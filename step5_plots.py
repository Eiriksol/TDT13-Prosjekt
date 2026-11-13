# python step5_plots.py <run name>, e.g. lda_k10_test_summary_s0
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

run = sys.argv[1]
predictions = pd.read_csv(f"results/predictions/{run}.csv")
data = pd.read_csv("data/conversations.csv", usecols=["convo_id", "flow"])
predictions = predictions.merge(data, on="convo_id")

counts = pd.crosstab(predictions["cluster"].astype(str), predictions["flow"])
# clusters dominated by the same flow are placed next to each other
order = sorted(counts.index, key=lambda cluster: counts.loc[cluster].argmax())
counts = counts.loc[order]
sizes = counts.sum(axis=1)

height = max(4, 0.25 * len(counts) + 2)
figure, (left, right) = plt.subplots(1, 2, figsize=(13, height), sharey=True, width_ratios=[1, 2])

left.barh(sizes.index, sizes.values, color="#3b6ea5")
left.invert_yaxis()
left.set_title("Conversations per cluster")
left.spines[["top", "right"]].set_visible(False)

right.imshow(counts.values, cmap="Blues", aspect="auto")
right.set_xticks(range(len(counts.columns)), counts.columns, rotation=45, ha="right")
right.set_title("Clusters vs. true flow")

figure.tight_layout()
Path("results/figures").mkdir(parents=True, exist_ok=True)
figure.savefig(f"results/figures/{run}.png", dpi=150)
print(f"saved results/figures/{run}.png")
