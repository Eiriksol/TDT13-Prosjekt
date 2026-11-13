from pathlib import Path

import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from helpers import get_args, load_texts, save_results


def get_embeddings(split, text_type, texts):
    # embeddings are slow on CPU, so they are only made once
    embedding_file = Path(f"data/embeddings/{split}_{text_type}.npy")
    if embedding_file.exists():
        return np.load(embedding_file)
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    embeddings = model.encode(["query: " + text for text in texts], batch_size=16, normalize_embeddings=True)
    embedding_file.parent.mkdir(exist_ok=True)
    np.save(embedding_file, embeddings)
    return embeddings


args = get_args()
_, train_texts = load_texts("train", args.text)
ids, texts = load_texts(args.split, args.text)
train_embeddings = get_embeddings("train", args.text, train_texts)
embeddings = get_embeddings(args.split, args.text, texts)

# learn the clusters on train
topic_model = BERTopic(
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=args.seed),
    hdbscan_model=KMeans(n_clusters=args.k, n_init=10, random_state=args.seed),
    vectorizer_model=CountVectorizer(stop_words="english"),
)
topic_model.fit(train_texts, train_embeddings)

# place each conversation in dev or test in the closest cluster
clusters, _ = topic_model.transform(texts, embeddings)

topics = {}
for topic, words in topic_model.get_topics().items():
    topics[int(topic)] = [word for word, _ in words]

save_results("bertopic", args, ids, clusters, {"topics": topics})
