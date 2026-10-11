# Contribution — Nilasu Yıldız

| | |
| --- | --- |
| Student ID | 2309011057 |
| GitHub | [@Xyzta](https://github.com/Xyzta) |
| Areas | Shared data preparation, evaluation, error analysis, documentation |

## Summary

I contributed to the shared data preparation and to the evaluation and documentation of the selected model. My work focused on held-out metrics, prediction exports, confusion matrices, error analysis, and clear reporting of limitations.

## Contributions

- Contributed to collecting and reviewing human-written and AI-generated texts used in the original corpus.
- Contributed to `pipeline/evaluation.py`, which calculates accuracy, macro precision, macro recall, and Macro-F1.
- Reviewed the external, HIP books, and Heterogeneous spans evaluation outputs.
- Contributed to the confusion matrices, prediction files, and misclassified-text exports in `outputs/latest/`.
- Contributed to `README.md`, `docs/benchmark.md`, `docs/data_provenance.md`, and the project limitations.
- Helped document that held-out datasets are evaluated only after training-CV model selection.

## Current final results

| Evaluation | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| External test | 0.8960 | 0.8960 |
| HIP books challenge | 0.7238 | 0.7217 |
| Heterogeneous spans challenge | 0.8610 | 0.8610 |

The lower HIP result is reported as evidence that human-like literary paraphrases are more difficult. The Heterogeneous result is reported separately because it measures partially AI-written documents from source groups excluded from training.

## Main files

| File | Purpose |
| --- | --- |
| `pipeline/evaluation.py` | Held-out metrics and error exports |
| `outputs/latest/` | Verified results and figures |
| `docs/benchmark.md` | Benchmark interpretation |
| `docs/data_provenance.md` | Data-source and split documentation |

Before submission, this summary should be checked against the contributor's actual GitHub commits and updated with commit or pull-request links where available.
