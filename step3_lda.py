from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from helpers import get_args, load_texts, save_results

args = get_args()
_, train_texts = load_texts("train", args.text)
ids, texts = load_texts(args.split, args.text)

# learn the topics on train
vectorizer = CountVectorizer(stop_words="english", min_df=2, max_df=0.5, token_pattern=r"\b[a-zA-Z]{3,}\b")
train_counts = vectorizer.fit_transform(train_texts)
lda = LatentDirichletAllocation(n_components=args.k, max_iter=50, random_state=args.seed)
lda.fit(train_counts)

# give each conversation in dev or test its most likely topic
word_counts = vectorizer.transform(texts)
topic_probabilities = lda.transform(word_counts)
clusters = topic_probabilities.argmax(axis=1)

words = vectorizer.get_feature_names_out()
topics = {}
for i, topic in enumerate(lda.components_):
    topics[i] = [str(words[j]) for j in topic.argsort()[::-1][:10]]

save_results("lda", args, ids, clusters, {"topics": topics})
