import json
import random
import urllib.request
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from helpers import get_args, load_texts, save_results

MODEL = "llama3.1:8b"
CONTEXT_LENGTH = 8192
CACHE_FILE = "data/llm_answers.jsonl"
GUIDELINES_URL = "https://github.com/asappresearch/abcd/raw/master/data/guidelines.json"
GUIDELINES_FILE = "data/guidelines.json"

SYSTEM = "You analyze customer service conversations for an online clothing store."

FIND_PROMPT = """Below are {n} customer service conversations.

{conversations}

List the distinct reasons customers contacted support in these conversations.
Use short snake_case names. Answer in JSON:
{{"categories": [{{"name": "...", "description": "..."}}]}}"""

MERGE_PROMPT = """Here are customer intent categories:

{categories}

Merge similar categories so that exactly {k} distinct categories remain, numbered 1 to {k}.
Put the most common categories first. Use short snake_case names and a one-sentence description.
Answer in JSON:
{{"categories": [{{"number": 1, "name": "...", "description": "..."}}]}}"""

RUNNING_PROMPT = """Categories found so far:
{categories}

Below are {n} customer service conversations.

{conversations}

For each conversation, give the reason the customer contacted support.
Use a category from the list when one fits. Only make a new short snake_case name if none fits, and then add a one-sentence description.
Answer in JSON, one entry per conversation:
{{"conversations": [{{"name": "...", "description": "..."}}]}}"""

ASSIGN_PROMPT ="""Categories:
{categories}

Conversation:
{conversation}

Which category best describes why the customer contacted support?
Answer in JSON: {{"number": <category number>}}"""

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
tokens = {"prompt": 0, "answer": 0}

# answers are saved, so a stopped run continues where it was
cache = {}
if Path(CACHE_FILE).exists():
    with open(CACHE_FILE, encoding="utf-8") as f:
        for line in f:
            saved = json.loads(line)
            cache[saved["prompt"]] = saved


def ask(prompt, temperature=0.0, max_tokens=1024):
    if prompt not in cache:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        saved = {
            "prompt": prompt,
            "answer": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "answer_tokens": response.usage.completion_tokens,
        }
        cache[prompt] = saved
        with open(CACHE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(saved) + "\n")

    saved = cache[prompt]
    tokens["prompt"] += saved["prompt_tokens"]
    tokens["answer"] += saved["answer_tokens"]
    if saved["prompt_tokens"] + max_tokens > CONTEXT_LENGTH:
        print("WARNING: prompt may be cut off,", saved["prompt_tokens"], "tokens")

    try:
        answer = json.loads(saved["answer"])
    except json.JSONDecodeError:
        return {}
    return answer if isinstance(answer, dict) else {}


def valid_categories(answer):
    # skips categories the model wrote in the wrong format, so one bad answer does not stop the run
    categories = answer.get("categories")
    if not isinstance(categories, list):
        return []
    return [c for c in categories if isinstance(c, dict) and isinstance(c.get("name"), str)]


def format_categories(categories):
    lines = []
    for i, category in enumerate(categories, start=1):
        lines.append(f"{i}. {category['name']}: {category.get('description', '')}")
    return "\n".join(lines)


def find_categories(texts, seed, n_batches=20, batch_size=20):
    rng = random.Random(seed)
    found = {}
    for _ in tqdm(range(n_batches), desc="find"):
        batch = rng.sample(texts, batch_size)
        conversations = "\n\n".join(f"[{i + 1}]\n{text[:1000]}" for i, text in enumerate(batch))
        answer = ask(FIND_PROMPT.format(n=batch_size, conversations=conversations), temperature=0.7)
        for category in valid_categories(answer):
            found[category["name"].lower()] = category
    return list(found.values())


def find_categories_running(texts, seed, batch_size=20, max_categories=200):
    # TopicGPT-style: goes through all of train and keeps a list that grows, with a count per category
    texts = random.Random(seed).sample(texts, len(texts))
    found = {}
    for start in tqdm(range(0, len(texts), batch_size), desc="find"):
        batch = texts[start:start + batch_size]
        # only the most used names are shown, so the prompt fits in the context window
        names = sorted(found, key=lambda name: -found[name]["count"])[:max_categories]
        conversations = "\n\n".join(f"[{i + 1}]\n{text[:1000]}" for i, text in enumerate(batch))
        prompt = RUNNING_PROMPT.format(categories="\n".join(names), n=len(batch), conversations=conversations)
        answer = ask(prompt, temperature=0.7, max_tokens=1500)
        for category in valid_categories({"categories": answer.get("conversations")}):
            name = category["name"].lower()
            if name not in found:
                found[name] = {"name": name, "description": category.get("description", ""), "count": 0}
            found[name]["count"] += 1
    found = sorted(found.values(), key=lambda category: -category["count"])[:max_categories]
    # the merge prompt shows the counts, so the model knows which categories are big
    return [{"name": c["name"], "description": f"{c['description']} (used {c['count']} times)"} for c in found]


def merge_categories(categories, k):
    # the model is bad at counting, so its own answer is merged again until it is small enough
    for round_number in range(1, 6):
        prompt = MERGE_PROMPT.format(categories=format_categories(categories), k=k)
        categories = valid_categories(ask(prompt, max_tokens=3000))
        print(f"merge round {round_number}: {len(categories)} categories")
        if len(categories) <= k:
            break
    return categories[:k]


def assign_category(text, categories):
    prompt = ASSIGN_PROMPT.format(categories=format_categories(categories), conversation=text[:4000])
    number = ask(prompt, max_tokens=64).get("number")
    if isinstance(number, int) and 1 <= number <= len(categories):
        return categories[number - 1]["name"]
    return "unassigned"


def flow_descriptions():
    # ABCD's guidelines have a one-line description per flow, e.g. "Product Defect": "refunds and returns"
    if not Path(GUIDELINES_FILE).exists():
        urllib.request.urlretrieve(GUIDELINES_URL, GUIDELINES_FILE)
    with open(GUIDELINES_FILE, encoding="utf-8") as f:
        guidelines = json.load(f)
    return {name.lower().replace(" ", "_").replace("-", "_"): flow["description"] for name, flow in guidelines.items()}


args = get_args()
_, train_texts = load_texts("train", args.text)
ids, texts = load_texts(args.split, args.text)

if args.gold:
    # not unsupervised: shows how well the LLM assigns when the categories are the true ones
    data = pd.read_csv("data/conversations.csv")
    # subflow names like "boots" only make sense with their flow, so they are written as flow/subflow
    names = data["flow"] if args.k == 10 else data["flow"] + "/" + data["subflow"]
    found = []
    categories = [{"name": name} for name in sorted(names.unique())]
    if args.gold == "descriptions":
        # only flows have short descriptions; subflows only have long agent instructions
        assert args.k == 10, "descriptions only exist for flows (k 10)"
        descriptions = flow_descriptions()
        for category in categories:
            category["description"] = descriptions[category["name"]]
else:
    # categories come from train, conversations in dev or test are assigned to them
    if args.find == "running":
        found = find_categories_running(train_texts, args.seed)
    else:
        found = find_categories(train_texts, args.seed)
    categories = merge_categories(found, args.k)
clusters = [assign_category(text, categories) for text in tqdm(texts, desc="assign")]

info = {
    "model": MODEL,
    "found": found,
    "categories": categories,
    "n_categories": len(categories),
    "n_unassigned": clusters.count("unassigned"),
    "tokens": tokens,
}
print("categories:", len(categories), "unassigned:", clusters.count("unassigned"))
method = {None: "llm", "names": "llmgold", "descriptions": "llmgolddesc"}[args.gold]
if args.find == "running":
    method = "llmrun"
save_results(method, args, ids, clusters, info)
