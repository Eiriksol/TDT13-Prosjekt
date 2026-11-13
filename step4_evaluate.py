import re
from glob import glob
from pathlib import Path

import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix


def clustering_accuracy(labels, clusters):
    # match each cluster to one label so that as many conversations as possible are right
    counts = contingency_matrix(labels, clusters)
    rows, columns = linear_sum_assignment(counts, maximize=True)
    return counts[rows, columns].sum() / counts.sum()


data = pd.read_csv("data/conversations.csv", usecols=["convo_id", "flow", "subflow"])

rows = []
for file in glob("results/predictions/*.csv"):
    run = Path(file).stem
    predictions = pd.read_csv(file).merge(data, on="convo_id")
    clusters = predictions["cluster"].astype(str)
    for level in ["flow", "subflow"]:
        labels = predictions[level]
        rows.append({
            "setup": re.sub(r"_s\d+$", "", run),
            "level": level,
            "ARI": adjusted_rand_score(labels, clusters),
            "AMI": adjusted_mutual_info_score(labels, clusters),
            "NMI": normalized_mutual_info_score(labels, clusters),
            "ACC": clustering_accuracy(labels, clusters),
            "clusters": clusters.nunique(),
        })

scores = pd.DataFrame(rows)
summary = scores.groupby(["setup", "level"]).agg(
    ARI=("ARI", "mean"),
    ARI_std=("ARI", "std"),
    AMI=("AMI", "mean"),
    NMI=("NMI", "mean"),
    ACC=("ACC", "mean"),
    ACC_std=("ACC", "std"),
    clusters=("clusters", "mean"),
    seeds=("ARI", "size"),
).round(3)

summary.to_csv("results/summary.csv")
print(summary.to_string())
