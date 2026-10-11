# Raw datasets

This directory contains the three datasets used in the final experiment. The project uses `0 = AI` and `1 = Human` in every file.

## AI vs Human Training Set.csv

**Purpose:** train and select the classifier with broad topics, writing styles, and AI-generation patterns.

**Sources and construction:**

- **Original project corpus:** 3,402 texts collected by the team. Human examples include course-book passages, student-style essays, comments, and personal or daily-language writing. AI texts were created on comparable topics so the model would not rely only on subject matter.
- **[HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3):** 1,000 matched human–ChatGPT pairs, or 2,000 texts. The pairs cover Reddit ELI5, finance, medicine, computer-science/Wikipedia, and open-domain questions. Each pair answers the same question, and texts were limited to approximately 50–500 words.
- **[Heterogeneous AI Spans](https://huggingface.co/datasets/open-text-detector/heterogeneous-ai-spans):** all 3,509 matched pairs from its training split, or 7,018 texts. Sources include Project Gutenberg, Standard Ebooks, WikiText-2, Federal Reserve Beige Book reports, pre-2015 JMLR papers, and historical Hansard. Each human control is paired with a version containing verified AI-written spans.

**Dataset characteristics:**

- 12,420 raw texts: 6,227 AI and 6,193 Human.
- The HC3 and Heterogeneous examples are stored as adjacent matched pairs.
- Mixed Heterogeneous documents are labeled AI because they contain AI-written spans; their recorded AI proportion ranges from approximately 4.3% to 88.8%.
- Pair-aware cross-validation keeps matched human and AI texts in the same fold.
- After the 50-word filter, 12,383 texts remain for training.

## AI vs Human Test Set.csv

**Purpose:** provide a balanced external evaluation set that is not used for feature fitting, model selection, or parameter tuning.

**Sources and construction:**

- **Original project test set:** 200 texts, containing 100 human student essays and 100 topic-matched AI texts. The human essays were selected from the [Kaggle LLM Detect AI Generated Text competition data](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data).
- **HC3 held-out extension:** 400 previously unused human–ChatGPT pairs, or 800 texts. It contains 100 pairs from each of Reddit ELI5, finance, medicine, and computer-science/Wikipedia.

**Dataset characteristics:**

- 1,000 texts: 500 AI and 500 Human.
- HC3 questions used for this file do not occur in the HC3 training selection.
- Exact normalized text overlap with the training and edge-case sets is zero.
- Columns are `id`, `prompt_id`, `text`, and `generated`. In this project, `generated = 0` means AI and `generated = 1` means Human. This is intentionally different from the original Kaggle label convention.

## AI vs Human Test Set (edge cases).csv

**Purpose:** measure robustness on difficult literary examples where AI writing is a paraphrase of a human passage rather than a simple standalone generation.

**Source and construction:**

- Selected from the books domain of the [HIP Training and Evaluation Data](https://huggingface.co/datasets/YixuanEvenXu/HIP-training-and-evaluation-data).
- The source provides original human passages and AI-generated paraphrases.
- We selected 400 unique pairs with high semantic preservation scores, comparable lengths, and no overlap with the training or standard test set.

**Dataset characteristics:**

- 800 texts: 400 AI paraphrases and 400 human originals.
- Every AI example has a human control covering the same content.
- The paired design reduces topic-based shortcuts and tests whether the classifier recognizes less obvious AI signals.
- This set is reported separately from the standard test set and is never used for tuning.

## Data handling rules

- Keep the three datasets separate.
- Use only the training set for model selection and cross-validation.
- Use the two test sets only for final evaluation.
- Verify the original source licenses and attribution requirements before redistributing the text. The repository software license does not replace dataset-specific terms.
