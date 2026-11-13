# upper bound: uses the labels, trained on train and predicts on test or dev
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from helpers import get_args, save_results

args = get_args()
level = {10: "flow", 55: "subflow"}[args.k]

data = pd.read_csv("data/conversations.csv", keep_default_na=False)
if args.text == "summary":
    data = data.merge(pd.read_csv("data/summaries.csv"), on="convo_id")
train = data[data["split"] == "train"]
test = data[data["split"] == args.split]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train = vectorizer.fit_transform(train[args.text])
X_test = vectorizer.transform(test[args.text])

model = LogisticRegression(max_iter=2000, C=10)
model.fit(X_train, train[level])
clusters = model.predict(X_test)

save_results("supervised", args, test["convo_id"].tolist(), clusters, {})
