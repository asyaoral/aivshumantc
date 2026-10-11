# Short presentation notes and Q&A

## Problem and data

This project classifies English texts as AI-generated or human-written. The final training dataset contains 12,420 texts before filtering. We also use a separate 1,000-text external test and an 800-text HIP books challenge set.

## Method

We compare Naive Bayes, Logistic Regression, and Linear SVM with word TF-IDF. We also test stop-word removal and six linguistic features. Pair-aware five-fold cross-validation keeps matched human and AI texts in the same fold.

## Results

Linear SVM with TF-IDF and six linguistic features achieved the best training CV macro-F1: 0.8961. It reached 0.8980 macro-F1 on the external test and 0.7391 on the HIP books challenge.

## Conclusion

The model performs well on the standard external test but is less reliable on difficult literary and partially modified texts. This difference shows why one high score is not enough to claim general AI-text detection.

## Short Q&A

### Why use macro-F1?

Macro-F1 gives equal importance to AI and human texts.

### How was leakage reduced?

TF-IDF and feature scaling are fitted inside each training fold. Matched pairs stay in the same fold. Exact overlap across datasets is rejected.

### Were the test sets used for model selection?

No. Model selection uses only training cross-validation. The external and HIP datasets are evaluated afterward.

### Why is the HIP score lower?

The HIP set contains more difficult literary and partially AI-modified texts. The AI signal may occur in only part of a long document.

### What is the main limitation?

The classifier may learn topic or source differences in addition to writing style. Its predictions do not prove authorship.

### What should be tested next?

Future work should test larger word features, smaller character features, limited SVM tuning, and document segmentation one step at a time. Final changes should be confirmed on a new untouched test set.
