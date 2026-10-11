"""End-to-end orchestration for the final three-dataset experiment."""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import (
    CHALLENGE_DATA_PATH,
    MIN_REQUIRED,
    OUTPUT_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from .data import clean_dataset, load_dataset, training_group, verify_no_overlap
from .evaluation import evaluate_holdout
from .models import evaluate_candidates, make_model


def _plot_dataset_overview(train, test, challenge, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    datasets = [
        (train, axes[0, 0], "Training class distribution"),
        (test, axes[0, 1], "External test distribution"),
        (challenge, axes[1, 0], "HIP challenge distribution"),
    ]
    for dataset, axis, title in datasets:
        dataset["Class"].map({0: "AI", 1: "Human"}).value_counts().reindex(
            ["AI", "Human"]
        ).plot.bar(ax=axis, rot=0)
        axis.set_title(title)
        axis.set_ylabel("Number of texts")
    axes[1, 1].hist(train["word_count"], bins=25, alpha=0.55, label="Training")
    axes[1, 1].hist(test["word_count"], bins=15, alpha=0.55, label="External test")
    axes[1, 1].hist(challenge["word_count"], bins=15, alpha=0.55, label="Challenge")
    axes[1, 1].set_title("Word count distribution")
    axes[1, 1].set_xlabel("Words")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_overview.png", dpi=160)
    plt.close(fig)


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [TRAIN_DATA_PATH, TEST_DATA_PATH, CHALLENGE_DATA_PATH]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required CSV files were not found: " + ", ".join(map(str, missing))
        )

    raw_train = load_dataset(TRAIN_DATA_PATH)
    raw_test = load_dataset(TEST_DATA_PATH)
    raw_challenge = load_dataset(CHALLENGE_DATA_PATH)
    train, train_quality = clean_dataset(raw_train, "Training")
    test, test_quality = clean_dataset(raw_test, "External test")
    challenge, challenge_quality = clean_dataset(raw_challenge, "HIP books challenge")
    verify_no_overlap(train, test, challenge)

    quality = pd.DataFrame([train_quality, test_quality, challenge_quality])
    quality.to_csv(output_dir / "data_quality.csv", index=False)
    print(quality.to_string(index=False))
    if len(train) < MIN_REQUIRED:
        warnings.warn(f"Only {len(train)} usable training samples; {MIN_REQUIRED} required.")

    _plot_dataset_overview(train, test, challenge, output_dir)

    X_train = train[["Text"]]
    y_train = train["Class"]
    groups = train["source_row"].map(training_group)
    candidates = evaluate_candidates(X_train, y_train, groups)
    candidates.to_csv(output_dir / "cv_model_comparison.csv", index=False)
    print("\nTRAINING CV RESULTS")
    print(candidates.to_string(index=False))

    plot_data = candidates.sort_values("CV F1 (macro)")
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.barh(plot_data["Experiment"], plot_data["CV F1 (macro)"])
    axis.set_xlabel("Mean 5-fold training CV F1 (macro)")
    axis.set_xlim(0, 1)
    axis.set_title("Model and feature comparison (external test untouched)")
    fig.tight_layout()
    fig.savefig(output_dir / "cv_comparison.png", dpi=160)
    plt.close(fig)

    selected = candidates.iloc[0]
    final_model = make_model(
        selected["Model"],
        remove_stop_words=bool(selected["Remove stop words"]),
        add_manual_features=bool(selected["Manual features"]),
    )
    final_model.fit(X_train, y_train)

    external_metrics, external_errors = evaluate_holdout(
        final_model,
        selected,
        "External test",
        "external_test",
        test[["Text"]],
        test["Class"],
        output_dir,
    )
    challenge_metrics, challenge_errors = evaluate_holdout(
        final_model,
        selected,
        "HIP books challenge",
        "hip_books_challenge",
        challenge[["Text"]],
        challenge["Class"],
        output_dir,
    )
    pd.DataFrame([external_metrics, challenge_metrics]).to_csv(
        output_dir / "final_holdout_results.csv", index=False
    )
    print(f"\nExternal-test errors: {len(external_errors)} / {len(test)}")
    print(f"HIP-challenge errors: {len(challenge_errors)} / {len(challenge)}")


if __name__ == "__main__":
    main()
