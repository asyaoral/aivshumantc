# Contribution — Asya Oral

| | |
| --- | --- |
| Student ID | 2309011045 |
| GitHub | [@asyaoral](https://github.com/asyaoral) |
| Areas | Shared data preparation, coding, model training, documentation |

## Summary

I contributed to the shared data preparation and to the reusable classification pipeline. My work focused on feature extraction, model construction, training cross-validation, and technical documentation.

## Contributions

- Contributed to collecting and reviewing human-written and AI-generated texts used in the original project corpus.
- Contributed to `pipeline/features.py`, which computes six deterministic linguistic features.
- Contributed to `pipeline/models.py`, which builds the TF-IDF, Logistic Regression, Linear SVM, and Multinomial Naive Bayes configurations.
- Contributed to the pair-aware five-fold cross-validation workflow and model comparison.
- Contributed to `pipeline/run.py`, `README.md`, and the technical documentation.
- Helped document why learned transformations remain inside scikit-learn pipelines so that validation-fold information does not enter feature fitting.

## Current experiment covered by the repository

- 11,383 usable training texts after the 50-word filter.
- Seven predefined model and feature configurations.
- Model selection by pair-aware five-fold Macro-F1.
- Final evaluation on the external, HIP books, and Heterogeneous spans datasets.

## Main files

| File | Purpose |
| --- | --- |
| `pipeline/features.py` | Six linguistic features |
| `pipeline/models.py` | Model pipelines and training CV comparison |
| `pipeline/run.py` | End-to-end experiment orchestration |
| `README.md` | Method and project documentation |

Before submission, this summary should be checked against the contributor's actual GitHub commits and updated with commit or pull-request links where available.
