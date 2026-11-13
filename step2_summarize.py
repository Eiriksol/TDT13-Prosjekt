# python step2_summarize.py --split all --max-hours 8.5
import argparse
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

MODEL = "llama3.1:8b"
SUMMARY_FILE = "data/summaries.csv"
PROMPT = "Summarize this customer service conversation in 2-3 sentences. Call the customer \"the customer\", not by name. Write only the summary.\n\n{conversation}"

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="dev", choices=["train", "dev", "test", "all"])
parser.add_argument("--max-hours", type=float, default=8.5)
args = parser.parse_args()

data = pd.read_csv("data/conversations.csv", keep_default_na=False)
if args.split == "all":
    data = data.sort_values("split")  # dev, test, then train
else:
    data = data[data["split"] == args.split]

# summaries are saved one by one, so a stopped run continues where it was
done = set()
if Path(SUMMARY_FILE).exists():
    done = set(pd.read_csv(SUMMARY_FILE)["convo_id"])

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
start = time.time()

for _, row in tqdm(data.iterrows(), total=len(data)):
    if row["convo_id"] in done:
        continue
    # stop before Kaggle ends the session, so the output is saved
    if time.time() - start > args.max_hours * 3600:
        print("stopped at the time limit, run again to continue")
        break
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.format(conversation=row["full"])}],
        temperature=0.0,
        max_tokens=200,
    )
    saved = {
        "convo_id": row["convo_id"],
        "summary": response.choices[0].message.content,
        "prompt_tokens": response.usage.prompt_tokens,
        "answer_tokens": response.usage.completion_tokens,
    }
    pd.DataFrame([saved]).to_csv(SUMMARY_FILE, mode="a", header=not Path(SUMMARY_FILE).exists(), index=False)
