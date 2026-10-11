# Contribution — Nilasu Yıldız

| | |
| --- | --- |
| Student ID | 2309011057 |
| GitHub | [@Xyzta](https://github.com/Xyzta) |
| Areas | Data preparation (shared), **evaluation**, **error analysis**, **documentation** |

## Summary
I took part in the team's shared data preparation. I was responsible for the **evaluation** of the selected model on the untouched holdout set and for the **error analysis**. I also wrote the project **documentation**: the main README, the result tables and the licenses.

## 1. Data preparation (shared with Asya Oral and Beyza Eser)
- Collected human-written English passages from books, articles and similar published sources (label 1).
- Generated AI texts on comparable topics with several different AI models (label 0).
- Labelled the texts and combined them into one table (`Text`, `Class`), kept in Apple Numbers and exported to CSV.
- Checked class balance and manually looked for empty, very short or duplicated texts.

## 2. Evaluation (`src/train.py`, evaluation part; `outputs/`)
- Evaluated the selected model **once** on the holdout set, after the model had been chosen by training CV only, so the holdout plays no part in model selection.
- Metrics: accuracy, macro precision, macro recall, macro-F1 and a per-class classification report. **Macro**-averaging gives the AI and Human classes equal weight.
- Figures and tables: `cv_comparison.png` (CV macro-F1 with SD error bars), `holdout_confusion_matrix.png`, `cv_model_comparison.csv`, `final_holdout_result.csv`.

**Main corpus result:** Linear SVM (TF-IDF, stop words kept)

| Holdout texts | Accuracy | Macro precision | Macro recall | Macro-F1 | Errors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 677 (340 AI / 337 Human) | 0.9823 | 0.9823 | 0.9823 | 0.9823 | 12 |

**Diagnostic set (200 texts):** all configurations scored 1.000 in CV and on the holdout. I reported this separately, because 200 texts is below the 1,000-sample requirement and a perfect score on such a small set is not evidence of general performance.

## 3. Error analysis
- Exported every holdout error to `misclassified_texts.csv` with its actual and predicted label and the full text.
- Reviewed the errors by hand. Patterns to check: human texts with a formal or encyclopedic tone that were predicted as AI, and AI texts with a personal or narrative tone that were predicted as Human. These are hypotheses, not verified model explanations.
- Interpreted the model comparison:
  - SVM and LR beat Naive Bayes by about 0.09–0.10 macro-F1. NB's independence assumption fits poorly with correlated n-gram features.
  - **Removing stop words slightly lowers performance** (0.9896 → 0.9885). Function words carry stylistic signal.
  - **The six linguistic features add nothing** on top of TF-IDF (0.9885). TF-IDF already captures most of what they encode.
- Listed the limitations: possible topic or source correlation with the label, exact-match-only deduplication, and results valid only within this corpus.

## 4. Documentation
- **`README.md`:** project overview, the purpose of every folder and script, dataset tables, run instructions, experimental setup, results, limitations, team table and references.
- **`LICENSE`** (MIT, code) and **`DATA_LICENSE.md`** (CC BY 4.0 for the compilation, labels and AI-generated texts; the original copyright of the human-written excerpts is kept).

## Files I mainly worked on
| File | Work |
| --- | --- |
| `src/train.py` | Holdout metrics, confusion matrix, misclassified-text export |
| `outputs/` | Result tables and figures |
| `README.md`, `LICENSE`, `DATA_LICENSE.md` | Project documentation and licensing |

## Challenges and solutions
- **Keeping the holdout untouched:** selection is based only on training CV, and the holdout is evaluated once at the end. This is stated in both the code and the README.
- **Not over-interpreting high scores:** the near-perfect results are reported together with their limitations.
