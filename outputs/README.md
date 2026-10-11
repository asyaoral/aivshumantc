# Verified outputs

`outputs/latest/` contains the results of the final four-dataset run:

- `data_quality.csv`: dataset sizes and class counts.
- `cv_model_comparison.csv`: pair-aware five-fold training CV results.
- `final_holdout_results.csv`: external, HIP, and Heterogeneous challenge metrics.
- `*_predictions.csv`: predictions for each held-out dataset.
- `*_misclassified_texts.csv`: error-analysis records.
- `*.png`: dataset, CV, and confusion-matrix figures.

The selected SVM reached 89.60% accuracy on the external test, 72.38% on the HIP books challenge, and 86.10% on the Heterogeneous spans challenge.
