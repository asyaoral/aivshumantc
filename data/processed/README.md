# Processed training data

`AI_vs_Human_Text_Classification_processed.csv` is the cleaned version of the final training dataset.

Processing normalizes whitespace, removes texts shorter than 50 words, excludes conflicting labels, and removes exact duplicates. The file contains 12,383 texts: 6,206 AI (`0`) and 6,177 Human (`1`).

The verified data-quality summary is stored in `outputs/latest/data_quality.csv`.
