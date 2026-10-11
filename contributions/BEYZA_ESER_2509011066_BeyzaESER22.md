# Contribution — Beyza Eser

| | |
| --- | --- |
| Student ID | 2509011066 |
| GitHub | [@BeyzaESER22](https://github.com/BeyzaESER22) |
| Areas | Shared data preparation, preprocessing, model comparison, training |

## Summary

I contributed to the shared data preparation and worked on the preprocessing and experiment design used in the final pipeline. My work focused on creating a clean training input, preventing leakage, comparing baseline classifiers, and evaluating the effect of the minimum-word threshold.

## Contributions

- Contributed to collecting, labelling, and reviewing the original human and AI examples.
- Worked on `pipeline/data.py`, including column standardization, label validation, whitespace normalization, the 50-word filter, conflicting-label removal, duplicate removal, and overlap checks.
- Worked on pair grouping so that matched HC3 and Heterogeneous human–AI examples remain in the same cross-validation fold.
- Contributed to the seven model configurations in `pipeline/models.py` and the end-to-end execution in `pipeline/run.py`.
- Ran and reviewed the main 50-word experiment and the separate 100-word sensitivity experiment.
- Contributed to integrating the source-group-held-out Heterogeneous challenge into the final benchmark.

## Current benchmark

| Evaluation | Macro-F1 |
| --- | ---: |
| Pair-aware training CV | 0.8971 |
| External test | 0.8960 |
| HIP books challenge | 0.7217 |
| Heterogeneous spans challenge | 0.8610 |

## Main files

| File | Purpose |
| --- | --- |
| `pipeline/data.py` | Loading, cleaning, deduplication, overlap checks, and grouping |
| `pipeline/config.py` | Dataset paths and experiment settings |
| `pipeline/models.py` | Candidate classifiers and cross-validation |
| `pipeline/run.py` | Complete training and evaluation flow |
| `scripts/create_heterogeneous_challenge.py` | Reproducible source-group holdout construction |

Before submission, this summary should be checked against the contributor's actual GitHub commits and updated with commit or pull-request links where available.
