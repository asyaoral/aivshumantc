# Evaluation protocol

Five configurations are compared with pair-aware stratified five-fold cross-validation on the training dataset. TF-IDF and feature scaling are fitted inside each fold. Matched human and AI examples remain in the same fold.

The selected model is the configuration with the highest mean macro-F1. Macro-F1 gives equal importance to AI (`0`) and Human (`1`).

The external test and HIP books challenge sets are evaluated only after model selection. Their results are reported separately because they measure different forms of generalization.

## Verified results

| Evaluation | Accuracy | Macro-F1 |
|---|---:|---:|
| Training CV | 0.8961 | 0.8961 |
| External test | 0.8980 | 0.8980 |
| HIP books challenge | 0.7400 | 0.7391 |

The best configuration is Linear SVM with word TF-IDF and six linguistic features.

## Interpretation

The external test result measures performance on a separate balanced dataset. The HIP result measures robustness on more difficult literary and partially AI-modified texts. Neither score proves authorship for an individual document.

Topic, source, length, and writing style may still correlate with labels. Exact-match checks also cannot identify every paraphrase or near-duplicate.
