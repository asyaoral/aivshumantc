# Short presentation notes and Q&A

## Problem and data

This project classifies English texts as AI-generated or human-written. The final training dataset contains 11,420 texts before filtering. We use a separate 1,000-text external test, an 800-text HIP books challenge, and a 1,000-text Heterogeneous mixed-authorship challenge.

## Method

We compare seven predefined configurations. Multinomial Naive Bayes provides a fast probabilistic TF-IDF baseline. For both Logistic Regression and Linear SVM, we compare TF-IDF with stop words kept, TF-IDF with stop words removed, and TF-IDF with six linguistic features. Pair-aware five-fold cross-validation keeps matched human and AI texts in the same fold. Scaled linguistic features are not combined with Multinomial Naive Bayes because that model requires non-negative inputs.

## Results

Linear SVM with TF-IDF and six linguistic features achieved the best training CV macro-F1: 0.8971. The strongest Logistic Regression variant reached 0.8743, while the Naive Bayes baseline reached 0.6937. The selected SVM reached 0.8960 macro-F1 on the external test, 0.7217 on the HIP books challenge, and 0.8610 on the Heterogeneous spans challenge.

## Conclusion

The model performs well on the standard external test and reaches 0.8610 macro-F1 on held-out mixed-authorship source groups, but it is less reliable on difficult literary paraphrases. This difference shows why one high score is not enough to claim general AI-text detection.

## Short Q&A

### Why use macro-F1?

Macro-F1 gives equal importance to AI and human texts.

### Why include Naive Bayes?

Multinomial Naive Bayes is a standard, fast probabilistic baseline for sparse word features. Comparing it with Logistic Regression and Linear SVM shows whether the more flexible discriminative models provide a real improvement over a simple traditional text classifier.

### How was leakage reduced?

TF-IDF and feature scaling are fitted inside each training fold. Matched pairs stay in the same fold. Exact overlap across datasets is rejected.

### Were the test sets used for model selection?

No. Model selection uses only training cross-validation. The external, HIP, and Heterogeneous challenge datasets are evaluated afterward.

### Why is the HIP score lower?

The HIP set contains difficult literary AI paraphrases that preserve the content of the matched human passages while changing the writing surface.

### How was the Heterogeneous challenge separated?

Five hundred matched pairs were removed from training. Splitting used `group_id`, so every passage from the same source document, book-author group, or paper remains entirely in training or entirely in the challenge set.

### What is the main limitation?

The classifier may learn topic or source differences in addition to writing style. Its predictions do not prove authorship.

### What should be tested next?

Future work should test larger word features, smaller character features, limited SVM tuning, and document segmentation one step at a time. Final changes should be confirmed on a new untouched test set.
