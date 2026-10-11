"""Train/validate five configurations; evaluate held-out data ONCE.

Run from repository root:
    python -m src.train --data data/raw/main.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .models import build_pipeline, candidate_configs, RANDOM_STATE
from .preprocess import clean_dataset, read_dataset


def train_and_evaluate(data_path: str | Path, output_dir: str | Path,
                       min_words: int = 50, require_samples: int = 1000,
                       folds: int = 5) -> dict:
    """One deterministic run. All model selection uses training-fold CV only."""
    path = Path(data_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = read_dataset(path)
    data, quality = clean_dataset(raw, min_words=min_words)
    quality["csv_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    if len(data) < require_samples:
        raise ValueError(
            f"Only {len(data)} clean examples; course requires {require_samples}. "
            "Do not submit this dataset. For temporary smoke testing only, pass --allow-small."
        )
    if data["Class"].nunique() != 2 or data["Class"].value_counts().min() < folds * 2:
        raise ValueError("Not enough data per class for stratified split and cross-validation.")

    # Explicit cleaned data output; do not publish until source and copyright permissions are verified.
    clean_path = Path("data/processed/cleaned.csv")
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(clean_path, index=False)

    X = data[["Text"]]
    y = data["Class"]
    train_indices, test_indices = train_test_split(
        np.arange(len(y)), test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    fold_plan = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}

    rows = []
    for config in candidate_configs():
        clf = build_pipeline(
            config["model"], remove_stop_words=config["remove_stop_words"],
            manual_features=config["manual_features"]
        )
        result = cross_validate(clf, X_train, y_train, cv=fold_plan,
                                scoring=scoring, n_jobs=1, error_score="raise")
        f1_folds = result["test_f1_macro"]
        rows.append({
            **config,
            "cv_accuracy_mean": float(np.mean(result["test_accuracy"])),
            "cv_f1_macro_mean": float(np.mean(f1_folds)),
            "cv_f1_macro_std": float(np.std(f1_folds)),
            "cv_f1_per_fold": json.dumps([round(float(z), 6) for z in f1_folds]),
        })
    # Stable tie-breaking by candidate order; DO NOT use holdout to select hyperparameters.
    leaderboard = pd.DataFrame(rows).sort_values("cv_f1_macro_mean", ascending=False, kind="stable")
    leaderboard.to_csv(out / "cv_benchmark.csv", index=False)
    best = leaderboard.iloc[0].to_dict()
    clf = build_pipeline(best["model"], remove_stop_words=bool(best["remove_stop_words"]),
                         manual_features=bool(best["manual_features"]))
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    metric_values = {
        "dataset_sha256": quality["csv_sha256"],
        "n_train": len(y_train), "n_holdout": len(y_test),
        "selected_model": best["name"],
        "selected_cv_f1_macro": float(best["cv_f1_macro_mean"]),
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_precision": float(precision_score(y_test, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "holdout_errors": int(np.sum(preds != y_test.to_numpy())),
    }
    pd.DataFrame([metric_values]).to_csv(out / "holdout_metrics.csv", index=False)
    detailed = classification_report(y_test, preds, labels=[0, 1],
                                     target_names=["AI", "Human"], zero_division=0, output_dict=True)
    (out / "classification_report.json").write_text(json.dumps(detailed, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["AI", "Human"]).plot(cmap="Blues", colorbar=False)
    plt.title("Untouched holdout • selected model")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close()

    plot = leaderboard.sort_values("cv_f1_macro_mean")
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    ax.barh(plot["name"], plot["cv_f1_macro_mean"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean macro-F1 (training 5-fold CV)")
    fig.tight_layout()
    fig.savefig(out / "cv_benchmark.png", dpi=160)
    plt.close(fig)

    # Do not commit full passages from books etc. without permission. These are locally ignored.
    err = pd.DataFrame({"Text": X_test["Text"].values,
                        "Actual": y_test.to_numpy(), "Predicted": preds})
    err = err.loc[err["Actual"].ne(err["Predicted"])].copy()
    private_dir = out / "private"
    private_dir.mkdir(exist_ok=True)
    err.to_csv(private_dir / "misclassified_texts_private.csv", index=False)
    all_error_types = pd.DataFrame([
        {"Actual": "AI", "Predicted": "Human", "Count": int(cm[0, 1])},
        {"Actual": "Human", "Predicted": "AI", "Count": int(cm[1, 0])},
    ])
    all_error_types.to_csv(out / "error_summary.csv", index=False)
    joblib.dump(clf, out / "selected_model.joblib")
    print("\nDATA QUALITY:", json.dumps({k: v for k, v in quality.items() if k != "csv_sha256"}, indent=2))
    print("\n5-FOLD CV BENCHMARK (TRAIN ONLY):")
    print(leaderboard[["name", "cv_accuracy_mean", "cv_f1_macro_mean", "cv_f1_macro_std"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFINAL HELD-OUT TEST (after CV selection):")
    print(json.dumps(metric_values, indent=2))
    print("\nOutputs:", out.resolve())
    return {"quality": quality, "leaderboard": leaderboard, "holdout": metric_values}


def main() -> None:
    p = argparse.ArgumentParser(description="Reproducible TF-IDF benchmark for AI vs Human text")
    p.add_argument("--data", default="data/raw/ai_vs_human.csv", help="Apple Numbers CSV export")
    p.add_argument("--output", default="outputs/latest")
    p.add_argument("--min-words", type=int, default=50)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--allow-small", action="store_true", help="For smoke tests only; not course submission")
    args = p.parse_args()
    train_and_evaluate(args.data, args.output, min_words=args.min_words,
                       require_samples=0 if args.allow_small else 1000, folds=args.folds)


if __name__ == "__main__":
    main()
