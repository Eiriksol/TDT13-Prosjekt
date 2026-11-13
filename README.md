# TDT13: Finding intents in customer service chats

Compares LDA, BERTopic and an LLM (Llama 3.1 8B) for grouping conversations in the [ABCD dataset](https://github.com/asappresearch/abcd) by contact reason, using either the raw dialogue or LLM summaries as input. Labels are only used for evaluation.

```
pip install -r requirements.txt

python step1_preprocess.py                                  # downloads ABCD, writes data/conversations.csv
python step3_lda.py --k 10 --split test --text summary --seed 0
python step3_bertopic.py --k 55 --split test --text full --seed 0
python step4_evaluate.py                                    # results/summary.csv
python step5_plots.py lda_k10_test_summary_s0               # results/figures/
```

`--k` is 10 (flows) or 55 (subflows), `--text` is full or summary. `step3_random.py` and `step3_supervised.py` are lower and upper bounds, and `notebooks/eda.ipynb` has dataset plots.

The LLM parts need a GPU and were run on Kaggle: `notebooks/kaggle_summarize.ipynb` makes the summaries (included as `data/summaries.csv`) and `notebooks/kaggle_llm.ipynb` runs all LLM experiments. `step3_llm.py` also takes `--find running` (go through all of train instead of 400 conversations) and `--gold names|descriptions` (use the true labels as categories, for comparison).

## AI Declariation

AI was used for help with writing and debugging code, for verifying the results, and for cleaning up the code before delivery.
