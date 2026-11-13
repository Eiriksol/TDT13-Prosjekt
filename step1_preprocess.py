import gzip
import json
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz"
RAW_FILE = "data/abcd_v1.1.json.gz"


def get_subflow(flow, subflow):
    # the data splits the FAQ subflows into questions (boots_how_3 -> boots), giving 96 instead of 55
    if flow in ["single_item_query", "storewide_query"]:
        return subflow.split("_")[0]
    # only 3 conversations, not a subflow in the ABCD guidelines
    if subflow == "status_delivery_date":
        return "status_delivery_time"
    return subflow


Path("data").mkdir(exist_ok=True)
if not Path(RAW_FILE).exists():
    urllib.request.urlretrieve(URL, RAW_FILE)

with gzip.open(RAW_FILE, "rt", encoding="utf-8") as f:
    raw = json.load(f)

rows = []
for split in ["train", "dev", "test"]:
    for convo in raw[split]:
        flow = convo["scenario"]["flow"]
        subflow = convo["scenario"]["subflow"]

        # action turns are system messages that reveal the label, so they are left out
        full = [f"{speaker}: {text}" for speaker, text in convo["original"] if speaker != "action"]

        rows.append({
            "split": split,
            "convo_id": convo["convo_id"],
            "flow": flow,
            "subflow": get_subflow(flow, subflow),
            "full": "\n".join(full),
        })

data = pd.DataFrame(rows)
data.to_csv("data/conversations.csv", index=False)

# same data, but easy to read
with open("data/conversations.txt", "w", encoding="utf-8") as f:
    for _, row in data.sort_values(["flow", "subflow"]).iterrows():
        f.write(f"=== {row['convo_id']} | {row['split']} | {row['flow']} / {row['subflow']} ===\n")
        f.write(row["full"] + "\n\n")

print(data["split"].value_counts())
print("flows:", data["flow"].nunique(), "subflows:", data["subflow"].nunique())
