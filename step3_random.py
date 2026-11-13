import numpy as np

from helpers import get_args, load_texts, save_results

args = get_args()
ids, texts = load_texts(args.split, args.text)

clusters = np.random.default_rng(args.seed).integers(args.k, size=len(texts))

save_results("random", args, ids, clusters, {})
