import argparse
import json
from pathlib import Path

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--text", default="full", choices=["full", "summary"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gold", choices=["names", "descriptions"])  # LLM only: the true labels as categories
    parser.add_argument("--find", default="sample", choices=["sample", "running"])  # LLM only: 400 conversations, or all of train with a running list
    return parser.parse_args()


def load_texts(split, text):
    data = pd.read_csv("data/conversations.csv", keep_default_na=False)
    if text == "summary":
        data = data.merge(pd.read_csv("data/summaries.csv"), on="convo_id")
    data = data[data["split"] == split]
    return data["convo_id"].tolist(), data[text].tolist()


def save_results(method, args, ids, clusters, info):
    name = f"{method}_k{args.k}_{args.split}_{args.text}_s{args.seed}"
    Path("results/predictions").mkdir(parents=True, exist_ok=True)
    Path("results/info").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"convo_id": ids, "cluster": clusters}).to_csv(f"results/predictions/{name}.csv", index=False)
    with open(f"results/info/{name}.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("saved", name)
