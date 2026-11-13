# Unsupervised Intent Discovery in Customer Service Conversations: Topic Models vs. LLMs

Code and results for the paper. Compares LDA, BERTopic and an LLM (Llama 3.1 8B) on the [ABCD dataset](https://github.com/asappresearch/abcd), with raw dialogue or LLM summaries as input.

## Results

All results are already in `results/`:
- `summary.csv`: every score in the paper
- `figures/`: dataset plots and heatmaps
- `info/`: topics and LLM categories for each run

Runs are named `method_k_split_text_seed`. `llmrun` is the LLM over all of train, `llmgold` and `llmgolddesc` use the true labels.

## Code

| File | Does |
|---|---|
| `step1_preprocess.py` | Downloads and cleans ABCD |
| `step2_summarize.py` | Summarizes conversations with the LLM |
| `step3_lda.py`, `step3_bertopic.py`, `step3_llm.py` | The three methods |
| `step3_random.py`, `step3_supervised.py` | Lower and upper bound |
| `step4_evaluate.py` | Scores all runs, writes `results/summary.csv` |
| `step5_plots.py` | Heatmap for one run |
| `notebooks/` | Dataset plots and the Kaggle notebooks used for the LLM |

## Run

```
pip install -r requirements.txt
python step1_preprocess.py
python step3_lda.py --k 10 --text summary
python step4_evaluate.py
```

`--k` is 10 (flows) or 55 (subflows), `--text` is full or summary. The summaries are included in `data/`. The LLM needs a GPU, but all its answers are saved in `data/llm_answers.jsonl.gz` (unzip to rerun without one).

## AI Declaration

AI was used for help with writing and debugging code, for verifying the results, and for cleaning up the code before delivery.
