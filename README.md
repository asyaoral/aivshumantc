# AI vs. Human Text Classification

This project classifies English texts as AI-generated (`0`) or human-written (`1`). It uses separate training, external test, and challenge datasets to measure both standard performance and robustness.

## Files

- `nlp_project1_ai_vs_human_clean.py`: small entry point that starts the pipeline.
- `pipeline/config.py`: paths and experiment settings.
- `pipeline/data.py`: loading, cleaning, overlap checks, and pair grouping.
- `pipeline/features.py`: six linguistic features.
- `pipeline/models.py`: TF-IDF pipelines and training cross-validation.
- `pipeline/evaluation.py`: held-out metrics, predictions, and figures.
- `pipeline/run.py`: end-to-end experiment flow.
- `data/raw/AI_vs_Human_Text_Classification_HC3_Heterogeneous_augmented.csv`: training dataset.
- `data/raw/Train_Data_Set_HC3_expanded.csv`: external test dataset.
- `data/raw/HIP_books_challenge_test.csv`: HIP books challenge dataset.
- `outputs/latest/`: verified metrics, predictions, error files, and figures.

## Dataset

The training dataset combines the original corpus with matched HC3 and Heterogeneous AI Spans examples. Human texts use label `1`; AI-generated texts use label `0`. Exact duplicates and cross-dataset overlaps were checked before evaluation.

| Dataset | Raw texts | Usable texts | AI | Human |
|---|---:|---:|---:|---:|
| Training | 12,420 | 12,383 | 6,206 | 6,177 |
| External test | 1,000 | 1,000 | 500 | 500 |
| HIP books challenge | 800 | 800 | 400 | 400 |

Texts shorter than 50 words are excluded. The 37 excluded texts occur only in the training dataset.

## Method

The code compares Multinomial Naive Bayes, Logistic Regression, and Linear SVM. It uses word-level TF-IDF with unigrams and bigrams and tests six linguistic features. Model selection is based only on pair-aware five-fold cross-validation within the training dataset. Matched human and AI texts remain in the same fold.

The external test and HIP challenge datasets are evaluated only after model selection. They are never used to fit the vocabulary, scale features, or choose the model.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

NLP_TRAIN_DATA_PATH="data/raw/AI_vs_Human_Text_Classification_HC3_Heterogeneous_augmented.csv" \
NLP_TEST_DATA_PATH="data/raw/Train_Data_Set_HC3_expanded.csv" \
NLP_CHALLENGE_DATA_PATH="data/raw/HIP_books_challenge_test.csv" \
NLP_OUTPUT_DIR="outputs/latest" \
python nlp_project1_ai_vs_human_clean.py
```

## Results

The selected model was Linear SVM with TF-IDF and six linguistic features.

| Evaluation | Accuracy | Macro-F1 |
|---|---:|---:|
| Pair-aware training CV | 89.61% | 89.61% |
| External test | 89.80% | 89.80% |
| HIP books challenge | 74.00% | 73.91% |

The lower challenge score shows that literary and partially modified texts are harder to classify. Results should not be interpreted as proof of authorship.

## Future Work and Improvement Guidelines

More expensive experiments were tested but not retained because combined word and character features created a cache larger than 6 GB and some SVM settings required excessive training time.

Future work should proceed incrementally:

1. Increase word TF-IDF to at most 20,000 features.
2. Test character TF-IDF separately with 10,000–15,000 features and 3–4 character n-grams.
3. Compare word and character models before combining them.
4. Test practical SVM values such as `C = 0.5` and `C = 1`.
5. Add linguistic features one group at a time.
6. Keep pair-aware cross-validation.
7. Segment long documents to detect small AI-modified regions.
8. Run larger experiments on a machine with more memory and processing power.

Model selection must use training-set cross-validation only. The external test and HIP challenge sets must not be used for tuning. Since the current external test has already been evaluated, future model changes should be confirmed on a new untouched test set.

## Limitations

- Topic and source differences may provide unintended clues to the classifier.
- Exact duplicate checks do not detect every paraphrase or near-duplicate.
- Source-level grouping is available only for known matched pairs.
- Dataset licenses and redistribution conditions must be verified before reuse.

Related work: [HC3](https://arxiv.org/abs/2301.07597), [SemEval-2024 Task 8](https://aclanthology.org/2024.semeval-1.148/), and [RAID](https://aclanthology.org/2024.acl-long.674/).
