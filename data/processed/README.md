# Processed training data

`AI_vs_Human_Text_Classification_processed.csv` is the cleaned version of the final training dataset.

Processing normalizes whitespace, removes texts shorter than 50 words, excludes conflicting labels, and removes exact duplicates. Five hundred matched Heterogeneous AI Spans pairs are reserved outside training as a source-group-held-out challenge set. The processed training file contains 11,383 texts: 5,706 AI or mixed-authorship texts (`0`) and 5,677 Human texts (`1`).

The verified data-quality summary is stored in `outputs/latest/data_quality.csv`.
