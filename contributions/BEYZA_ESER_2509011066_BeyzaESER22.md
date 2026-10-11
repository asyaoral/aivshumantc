# Contribution — Beyza Eser

| | |
| --- | --- |
| Student ID | 2509011066 |
| GitHub | [@BeyzaESER22](https://github.com/BeyzaESER22) |
| Areas | Data preparation (shared), **data preprocessing**, **models**, **training** |

## Summary
I took part in the team's shared data preparation. I was responsible for the **preprocessing pipeline**, which turns the raw collected texts into a clean, deduplicated dataset. I also chose the **classification models** and designed and ran the **training experiments** (cross-validation and model selection).

## 1. Data preparation (shared with Asya Oral and Nilasu Yıldız)
- Collected human-written English passages from books, articles and similar published sources (label 1).
- Generated AI texts on comparable topics with several different AI models (label 0), so the AI class is not tied to a single generator.
- Labelled the texts and combined them into one table (`Text`, `Class`), kept in Apple Numbers and exported to CSV.
- Checked class balance and manually looked for empty, very short or duplicated texts.

## 2. Data preprocessing: `src/preprocess.py`, `data/cleaned/`
- Loader for comma- or semicolon-separated CSV and `.numbers` files, with automatic detection of the text and label columns (`Text`; `Class` / `label` / `generated`).
- Whitespace normalization. Punctuation, casing and stop words are **kept** on purpose because they are stylistic signals.
- Label mapping (`0/1`, `AI/Human`, ...) to **0 = AI, 1 = Human**, which stops on any unknown label instead of continuing silently.
- Removal of texts under **50 words**.
- **Conflicting labels:** identical texts with different labels are removed as a whole group, because keeping one would mean guessing the label.
- **Exact (case-insensitive) duplicate removal,** so the same text cannot end up in both the training and holdout sets.
- `data_quality.csv` report with the number of rows removed at each step.
- Result: the main corpus went from 3,402 to **3,382** usable texts (1,698 AI / 1,684 Human; 20 short texts removed). The diagnostic set kept all 200 texts (100 / 100).

## 3. Models
Chose and configured the three classifiers that are compared:
- **Multinomial Naive Bayes** (course requirement): a probabilistic baseline for TF-IDF counts.
- **Logistic Regression** (`max_iter=2000`): a linear probabilistic model.
- **Linear SVM** (`LinearSVC`, `max_iter=10000`): a margin-based linear model, which is strong on high-dimensional sparse text.
- Shared text representation for all models: TF-IDF with unigrams and bigrams and at most 5,000 features.

## 4. Training: experiment design and runs (`src/train.py`)
- Stratified **80/20 train/holdout split** (`random_state=42`). The holdout is kept untouched.
- Stratified **5-fold cross-validation on the training split only**. Selection by mean **macro-F1**.
- Designed and ran the five experiments:

| Experiment | Mean CV macro-F1 | SD |
| --- | ---: | ---: |
| SVM, TF-IDF (stop words kept) | **0.9896** | 0.0022 |
| SVM, TF-IDF (stop words removed) | 0.9885 | 0.0036 |
| SVM, TF-IDF + 6 linguistic features | 0.9885 | 0.0049 |
| Logistic Regression, TF-IDF | 0.9830 | 0.0061 |
| Naive Bayes, TF-IDF | 0.8904 | 0.0179 |

- The selected configuration (SVM, TF-IDF, stop words kept) was refit on the full training split for the final evaluation.

## Files I mainly worked on
| File | Work |
| --- | --- |
| `src/preprocess.py` | Full preprocessing pipeline and command-line interface |
| `data/raw/`, `data/cleaned/` | Raw data organisation and cleaned outputs |
| `src/train.py` | Split, CV experiment list, model selection |

## Challenges and solutions
- **Different file formats:** the data came as `.numbers` and as CSV files with different separators. Format and column auto-detection lets one script handle all of them.
- **Fair comparison:** every model uses the same split, the same folds and the same TF-IDF settings, so the differences come only from the model or the feature choice.
