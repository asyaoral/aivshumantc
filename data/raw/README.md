# Raw datasets

All files use `0 = AI or mixed authorship` and `1 = Human`.

## Heterogeneous AI Spans split

The original `train-2.parquet` contains 3,509 matched pairs. Each `source_id` has one fully human control and one mixed-authorship version containing AI-written spans. The sources are Project Gutenberg, Standard Ebooks, JMLR papers, Federal Reserve Beige Book reports, WikiText-2, and Hansard.

Five hundred pairs from 123 complete `group_id` source groups were reserved in `AI vs Human Test Set (heterogeneous spans).csv`. No passage belonging to those source groups remains in `AI vs Human Training Set.csv`. The remaining 3,009 pairs stay in training. The split is reproducible with `scripts/create_heterogeneous_challenge.py` and `random_state = 18093`.

The Parquet file records construction-based span annotations, but `human_verified` is `False` for every row. Therefore, these spans should not be described as manually human-verified.

## Data handling rules

- Keep the training dataset and the three held-out datasets separate.
- Use only the training set for model selection and cross-validation.
- Use the external, HIP, and Heterogeneous challenge sets only for final evaluation.
- Verify the original source licenses and attribution requirements before redistributing the text. The repository software license does not replace dataset-specific terms.

## Reproducibility boundary

The original corpus, Kaggle essay selection, AI counterpart generation, and HC3 pair selection included manual collection and review. Their provenance and selection rules are documented here and in `docs/data_provenance.md`, but the repository does not claim that these early steps can be recreated automatically from a single command.

The committed raw CSV files are the input boundary for the automated pipeline. From that point onward, cleaning, label validation, duplicate and overlap checks, pair grouping, cross-validation, and evaluation are reproducible. The Heterogeneous source-group holdout can additionally be rebuilt from its original Parquet source with `scripts/create_heterogeneous_challenge.py`.

The CSV files contain material governed by different upstream terms. See `DATA_LICENSE.md` before reuse or redistribution.
