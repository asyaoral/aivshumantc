# Evaluation protocol and related work

## Internal benchmark

Five models/settings are compared on exactly the same 80% **training** partition using five-fold stratified CV. Each pipeline fits TF-IDF vocabulary and any scaler strictly within its own training fold. Choose the best candidate by **mean macro-F1**. Report cross-validation accuracy, mean macro-F1 and standard deviation. Evaluate the untouched 20% holdout **once**, after model selection, and show macro precision, macro recall, macro-F1, accuracy and confusion matrix.

Macro-F1 gives AI (`0`) and Human (`1`) equal importance and discourages misleadingly strong accuracy when one class dominates. We additionally report accuracy for readability; this particular corpus is approximately balanced.

Compare Multinomial Naive Bayes, Logistic Regression and Linear SVM as required class baselines. Compare stop-word ablation and the six hand-crafted linguistic features on SVM. Do not label a method not implemented in `src/models.py` as a new method.

## Related published datasets and robustness tests (QUALITATIVE only)

- HC3: Guo et al. (2023), *How Close is ChatGPT to Human Experts?* https://arxiv.org/abs/2301.07597
- SemEval-2024 Task 8: https://aclanthology.org/2024.semeval-1.148/
- RAID: Dugan et al. (2024), https://aclanthology.org/2024.acl-long.674/

The papers use different tasks, domains and testing procedures. Their raw accuracy cannot be ranked against ours. A fair numerical external benchmark requires the same evaluation dataset and protocol. Our present result is **within-corpus** generalization only, not real-world AI authorship proof.

## Limitations / data novelty

- Exact-match deduplication does not remove near-duplicates or texts paraphrased from the same source.
- Subject/source differences can leak dataset provenance: human book excerpts vs AI-generated texts may not match in topic.
- Validate that data augmentation preserved credible labels and copyright permissions. The 200 flipped-label essays are NOT part of our main clean dataset.
- For the professor's data novelty score, record original sources, transformations, new samples, and the cleaning script (`src/preprocess.py`) honestly.
