# Evaluation protocol

Seven predefined configurations are compared with pair-aware stratified five-fold cross-validation on the training dataset. The comparison includes a Multinomial Naive Bayes TF-IDF baseline and three variants each of Logistic Regression and Linear SVM: TF-IDF with stop words kept, TF-IDF with stop words removed, and TF-IDF with six linguistic features. Scaled linguistic features are not passed to Multinomial Naive Bayes because the scaler can produce negative values and that classifier requires non-negative inputs. TF-IDF and feature scaling are fitted inside each fold. Matched human and AI examples remain in the same fold.

The selected model is the configuration with the highest mean macro-F1. Macro-F1 gives equal importance to AI (`0`) and Human (`1`).

| Candidate configuration | Mean CV macro-F1 |
|---|---:|
| Linear SVM, TF-IDF + 6 linguistic features | 0.8971 |
| Linear SVM, TF-IDF, stop words kept | 0.8952 |
| Logistic Regression, TF-IDF, stop words kept | 0.8743 |
| Logistic Regression, TF-IDF + 6 linguistic features | 0.8719 |
| Linear SVM, TF-IDF, stop words removed | 0.8585 |
| Logistic Regression, TF-IDF, stop words removed | 0.8517 |
| Multinomial Naive Bayes, TF-IDF, stop words kept | 0.6937 |

The external test, HIP books challenge, and Heterogeneous spans challenge sets are evaluated only after model selection. Their results are reported separately because they measure different forms of generalization.

## Verified results

| Evaluation | Accuracy | Macro-F1 |
|---|---:|---:|
| Training CV | 0.8971 | 0.8971 |
| External test | 0.8960 | 0.8960 |
| HIP books challenge | 0.7238 | 0.7217 |
| Heterogeneous spans challenge | 0.8610 | 0.8610 |

The best configuration is Linear SVM with word TF-IDF and six linguistic features.

## Interpretation

The external test result measures performance on a separate balanced dataset. The HIP result measures robustness on literary AI paraphrases. The Heterogeneous result measures mixed-authorship detection using complete source groups excluded from training. None of these scores proves authorship for an individual document.

Topic, source, length, and writing style may still correlate with labels. Exact-match checks also cannot identify every paraphrase or near-duplicate.
