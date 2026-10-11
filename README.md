# AI vs. Human Text Classification

This project classifies English texts as AI-generated (`0`) or human-written (`1`). It uses separate training, external test, HIP challenge, and Heterogeneous AI Spans challenge datasets to measure both standard performance and robustness.

## Files

- `nlp_project1_ai_vs_human_clean.py`: small entry point that starts the pipeline.
- `pipeline/config.py`: paths and experiment settings.
- `pipeline/data.py`: loading, cleaning, overlap checks, and pair grouping.
- `pipeline/features.py`: six linguistic features.
- `pipeline/models.py`: TF-IDF pipelines and training cross-validation.
- `pipeline/evaluation.py`: held-out metrics, predictions, and figures.
- `pipeline/run.py`: end-to-end experiment flow.
- `data/raw/AI vs Human Training Set.csv`: training dataset.
- `data/raw/AI vs Human Test Set.csv`: external test dataset.
- `data/raw/AI vs Human Test Set (edge cases).csv`: HIP books challenge dataset.
- `data/raw/AI vs Human Test Set (heterogeneous spans).csv`: source-group-held-out mixed-authorship challenge dataset.
- `data/processed/AI_vs_Human_Text_Classification_processed.csv`: cleaned training dataset used by the pipeline.
- `outputs/latest/`: verified metrics, predictions, error files, and figures.
- `tests/`: automated checks for loading, cleaning, leakage detection, pair grouping, linguistic features, and model construction.
- `DATA_LICENSE.md`: source-specific dataset terms and third-party notices.

## Dataset

The training dataset combines the original corpus with matched HC3 and Heterogeneous AI Spans examples. Five hundred Heterogeneous pairs were removed from training and reserved as a source-group-held-out challenge set. Human texts use label `1`; AI-generated or mixed-authorship texts use label `0`. Exact duplicates and cross-dataset overlaps were checked before evaluation.

| Dataset | Raw texts | Usable texts | AI | Human |
|---|---:|---:|---:|---:|
| Training | 11,420 | 11,383 | 5,706 | 5,677 |
| External test | 1,000 | 1,000 | 500 | 500 |
| HIP books challenge | 800 | 800 | 400 | 400 |
| Heterogeneous spans challenge | 1,000 | 1,000 | 500 | 500 |


# Raw datasets

The project uses `0 = AI` and `1 = Human` in every file.

## AI vs Human Training Set.csv

**Purpose:** train and select the classifier with broad topics, writing styles, and AI-generation patterns.

**Sources and construction:**
- **Original project corpus:** 852 texts collected by the team. Human examples include course-book passages, student-style essays, comments, and personal or daily-language writing. AI texts were created on comparable topics so the model would not rely only on subject matter.
- **[Kaggle LLM Detect AI Generated Text competition data](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data):** 1,275 human essays were selected, and 1,275 AI texts were created on corresponding topics.
- **[HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3):** 1,000 matched human–ChatGPT pairs, or 2,000 texts. The pairs cover Reddit ELI5, finance, medicine, computer-science/Wikipedia, and open-domain questions. Each pair answers the same question, and texts were limited to approximately 50–500 words.
- **[Heterogeneous AI Spans](https://huggingface.co/datasets/open-text-detector/heterogeneous-ai-spans):** 3,009 matched pairs, or 6,018 texts, remain in training after 500 complete source-group-held-out pairs were reserved for the Heterogeneous challenge. Sources include Project Gutenberg, Standard Ebooks, WikiText-2, Federal Reserve Beige Book reports, pre-2015 JMLR papers, and historical Hansard. Each human control is paired with a mixed-authorship version containing construction-annotated AI spans.

**Dataset characteristics:**

- 11,420 raw texts: 5,727 AI or mixed-authorship texts and 5,693 Human texts.
- The HC3 and Heterogeneous examples are stored as adjacent matched pairs.
- Mixed Heterogeneous documents are labeled AI because they contain AI-written spans; their recorded AI proportion ranges from approximately 4.3% to 88.8%.
- Pair-aware cross-validation keeps matched human and AI texts in the same fold.
- After the 50-word filter, 11,383 texts remain for training.



## AI vs Human Test Set.csv

**Purpose:** provide a balanced external evaluation set that is not used for feature fitting, model selection, or parameter tuning.

**Sources and construction:**
- **[Kaggle LLM Detect AI Generated Text competition data](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data):** 100 human essays and 100 topic-matched AI texts were selected without reusing training examples.
- **HC3 held-out extension:** 400 previously unused human–ChatGPT pairs, or 800 texts. It contains 100 pairs from each of Reddit ELI5, finance, medicine, and computer-science/Wikipedia.

**Dataset characteristics:**
- 1,000 texts: 500 AI and 500 Human.
- HC3 questions used for this file do not occur in the HC3 training selection.
- Exact normalized text overlap with the training and edge-case sets is zero.
- Columns are `id`, `prompt_id`, `text`, and `generated`. In this project, `generated = 0` means AI and `generated = 1` means Human. This is intentionally different from the original Kaggle label convention.





## AI vs Human Test Set (edge cases).csv

**Purpose:** For evaulating the model performance on more diverse data set to avoid the lack of interperation sourced from model shortcut if the text is orderly and error free then it is AI; if it is personal and contains spelling errors then it is human. This data set contain AI generated text writen in a natural, human-like style or with human-written texts that employ academic language. measure robustness on difficult literary examples where AI writing is a paraphrase of a human passage rather than a simple standalone generation.

**Source and construction:**
- Selected from the books domain of the [HIP Training and Evaluation Data](https://huggingface.co/datasets/YixuanEvenXu/HIP-training-and-evaluation-data).
- The source provides original human passages and AI-generated paraphrases.
- We selected 400 unique pairs with high semantic preservation scores, comparable lengths, and no overlap with the training or standard test set.

**Dataset characteristics:**
- 800 texts: 400 AI paraphrases and 400 human originals.
- Every AI example has a human control covering the same content.
- The paired design reduces topic-based shortcuts and tests whether the classifier recognizes less obvious AI signals.
- This set is reported separately from the standard test set and is never used for tuning.




## AI vs Human Test Set (heterogeneous spans).csv

**Purpose:** measure performance on mixed-authorship documents whose complete source groups were excluded from training.

**Source and construction:**
- Created from the original 3,509 Heterogeneous AI Spans pairs.
- Contains 500 mixed-authorship texts and their 500 matched human controls.
- The 500 pairs belong to 123 complete `group_id` source groups.
- Every passage from a selected source group is held out, preventing the same book, paper, report, or source document from appearing in training.
- The split is reproducible with `scripts/create_heterogeneous_challenge.py`.

**Dataset characteristics:**
- 1,000 texts: 500 mixed-authorship (`0`) and 500 Human (`1`).
- AI character fractions in the original source range from approximately 4.3% to 88.8%.
- The original Parquet records construction-based span annotations; its `human_verified` field is `False` for all rows.


Texts shorter than 50 words are excluded. The 37 excluded texts occur only in the training dataset.

## Method

The code compares seven predefined configurations of Multinomial Naive Bayes, Logistic Regression, and Linear SVM. Naive Bayes is included as a fast probabilistic text-classification baseline, while Logistic Regression and Linear SVM provide stronger discriminative linear comparisons for high-dimensional sparse text. The models use word-level TF-IDF with unigrams and bigrams. Stop-word removal and the addition of six linguistic features are tested separately for Logistic Regression and Linear SVM. Scaled linguistic features are not combined with Multinomial Naive Bayes because scaling can produce negative values, whereas that classifier requires non-negative inputs. Model selection is based only on pair-aware five-fold cross-validation within the training dataset. Matched human and AI texts remain in the same fold.

The external test, HIP challenge, and Heterogeneous challenge datasets are evaluated only after model selection. They are never used to fit the vocabulary, scale features, or choose the model.

### Training-CV model comparison

The following values come from the current 11,383-text training dataset, after 500 Heterogeneous pairs were moved to the separate challenge set. All candidates use the same pair-aware five-fold splits, and the held-out datasets are not used in this comparison.

| Candidate configuration | Mean CV macro-F1 |
|---|---:|
| Linear SVM, TF-IDF + 6 linguistic features | **0.8971** |
| Linear SVM, TF-IDF, stop words kept | 0.8952 |
| Logistic Regression, TF-IDF, stop words kept | 0.8743 |
| Logistic Regression, TF-IDF + 6 linguistic features | 0.8719 |
| Linear SVM, TF-IDF, stop words removed | 0.8585 |
| Logistic Regression, TF-IDF, stop words removed | 0.8517 |
| Multinomial Naive Bayes, TF-IDF, stop words kept | 0.6937 |

Multinomial Naive Bayes is included as a fast, traditional probabilistic baseline for sparse text features. Its lower score shows that its simple conditional-independence assumptions are less suitable for this dataset than the discriminative linear models. Scaled linguistic features are not added to Multinomial Naive Bayes because `StandardScaler` can produce negative values, while `MultinomialNB` requires non-negative inputs.

Linear SVM is the strongest tested algorithm for this high-dimensional sparse TF-IDF representation. Keeping stop words improves both SVM and Logistic Regression, suggesting that common function words carry useful writing-style information. The six linguistic features produce only a small numerical improvement for SVM and slightly reduce Logistic Regression performance; therefore, their contribution should be described as modest rather than conclusive.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

NLP_TRAIN_DATA_PATH="data/raw/AI vs Human Training Set.csv" \
NLP_TEST_DATA_PATH="data/raw/AI vs Human Test Set.csv" \
NLP_CHALLENGE_DATA_PATH="data/raw/AI vs Human Test Set (edge cases).csv" \
NLP_HETEROGENEOUS_CHALLENGE_DATA_PATH="data/raw/AI vs Human Test Set (heterogeneous spans).csv" \
NLP_OUTPUT_DIR="outputs/latest" \
python nlp_project1_ai_vs_human_clean.py
```

The default minimum text length is 50 words. A separate sensitivity run can use a different threshold without changing the code or overwriting the main outputs:

```bash
NLP_MIN_WORDS=100 \
NLP_OUTPUT_DIR="outputs/min_words_100" \
python nlp_project1_ai_vs_human_clean.py
```

## Test

Run the automated checks before training or publishing the repository:

```bash
pytest -q
```

The tests cover delimiter and column handling, label validation, the 50-word rule, conflicting-label removal, exact duplicate removal, cross-dataset leakage detection, pair grouping, linguistic-feature output, and supported model construction.

## Reproducibility boundary

The repository fully reproduces cleaning, feature extraction, pair-aware cross-validation, model selection, evaluation, and the Heterogeneous source-group split. `scripts/create_heterogeneous_challenge.py` recreates that challenge from the original Parquet file.

The earliest collection and selection of the original corpus, Kaggle essays, and HC3 pairs included manual work and is documented in `data/raw/README.md` and `docs/data_provenance.md`; it is not presented as a fully automated download-to-CSV process. The committed raw CSV files are therefore the reproducible input boundary for the main pipeline.

Dataset content has mixed upstream terms. See `DATA_LICENSE.md` before reuse or redistribution; the MIT license applies only to team-authored software and documentation.

## Results

The selected model was Linear SVM with TF-IDF and six linguistic features.

| Evaluation | Accuracy | Macro-F1 |
|---|---:|---:|
| Pair-aware training CV | 89.71% | 89.71% |
| External test | 89.60% | 89.60% |
| HIP books challenge | 72.38% | 72.17% |
| Heterogeneous spans challenge | 86.10% | 86.10% |

The lower HIP score shows that literary paraphrases are particularly difficult to classify. The Heterogeneous result measures detection of mixed-authorship documents from source groups excluded from training. Results should not be interpreted as proof of authorship.

## Future Work and Improvement Guidelines

The current implementation uses word-level TF-IDF, six linguistic features, and pair-aware five-fold cross-validation. A larger experimental configuration was started but was not retained as the final pipeline because its combined computational cost was not practical on the available machine.

The attempted extension included:

- Increasing word-level TF-IDF from 5,000 to 20,000 features.
- Adding character-level TF-IDF features.
- Combining word and character representations.
- Expanding the linguistic feature set from six to twelve features.
- Testing multiple SVM regularization values.
- Building a voting ensemble with SVM and Logistic Regression.

With five-fold cross-validation, every configuration must be fitted five times. Combining the larger word matrix, character matrix, additional features, SVM settings, and ensemble members therefore multiplied both memory use and training time. The generated feature cache exceeded 6 GB. Higher SVM `C` configurations also ran for a long time and some did not converge within their iteration limit. The run was stopped before it produced a complete and fairly comparable result table. For that reason, no partial score from the interrupted experiment is reported as a project result; the stable and reproducible seven-configuration pipeline was retained.

### Evidence retained in the repository

| File | What it shows |
|---|---|
| [`pipeline/models.py`](pipeline/models.py) | The retained experiment is deliberately bounded to 5,000 word TF-IDF features and seven reproducible candidate configurations. It does not silently include an unfinished character model or ensemble. |
| [`outputs/latest/data_quality.csv`](outputs/latest/data_quality.csv) | The final run processes 11,383 usable training texts. This provides the dataset scale against which every five-fold candidate is fitted. |
| [`outputs/latest/cv_model_comparison.csv`](outputs/latest/cv_model_comparison.csv) | All seven retained candidates completed the same pair-aware five-fold evaluation. The best mean macro-F1 is 0.8971, so the selected model is supported by a complete comparison rather than a partial run. |
| [`outputs/latest/cv_comparison.png`](outputs/latest/cv_comparison.png) | A visual version of the completed candidate comparison. |
| [`outputs/latest/final_holdout_results.csv`](outputs/latest/final_holdout_results.csv) | The selected stable model reaches 0.8960 macro-F1 on the external test, 0.7217 on HIP, and 0.8610 on the Heterogeneous challenge. The gap between datasets supports treating representation and segmentation improvements as controlled future experiments. |

The temporary cache larger than 6 GB and incomplete non-convergence output were runtime artifacts, not valid experimental results. They were not committed because of their size and because the interrupted configurations did not finish under the same evaluation protocol. Therefore, the repository evidence documents the completed baseline and the reasoned scope decision; it should not be presented as if a full larger-model benchmark had been completed.

Future work should proceed incrementally:

1. Increase word TF-IDF to at most 20,000 features.
2. Test character TF-IDF separately with 10,000–15,000 features and 3–4 character n-grams.
3. Compare word and character models before combining them.
4. Test practical SVM values such as `C = 0.5` and `C = 1`.
5. Add linguistic features one group at a time.
6. Keep pair-aware cross-validation.
7. Segment long documents to detect small AI-modified regions.
8. Run larger experiments on a machine with more memory and processing power.

Model selection must use training-set cross-validation only. None of the three held-out sets may be used for tuning. Since these sets have now been evaluated, future model changes should be confirmed on a new untouched test set.

## Limitations

- Topic and source differences may provide unintended clues to the classifier.
- Exact duplicate checks do not detect every paraphrase or near-duplicate.
- Source-level grouping is available only for known matched pairs.
- Dataset licenses and redistribution conditions must be verified before reuse.

Related work: [HC3](https://arxiv.org/abs/2301.07597), [SemEval-2024 Task 8](https://aclanthology.org/2024.semeval-1.148/), and [RAID](https://aclanthology.org/2024.acl-long.674/).
