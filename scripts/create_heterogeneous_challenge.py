"""Create a source-group-held-out Heterogeneous AI Spans challenge set."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ORIGINAL_AND_HC3_ROWS = 5402
SELECTION_SEED = 18093
GROUP_TEST_SIZE = 0.143
EXPECTED_PAIRS = 500


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path, help="Original train-2.parquet file")
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=Path("data/raw/AI vs Human Training Set.csv"),
    )
    parser.add_argument(
        "--challenge-csv",
        type=Path,
        default=Path("data/raw/AI vs Human Test Set (heterogeneous spans).csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_parquet(args.parquet)
    required = {
        "source_id", "group_id", "source_dataset", "generator_backend",
        "kind", "text", "ai_fraction_chars",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Missing Parquet columns: {sorted(missing)}")

    pair_counts = source.groupby("source_id")["kind"].agg(["size", "nunique"])
    if not ((pair_counts["size"] == 2) & (pair_counts["nunique"] == 2)).all():
        raise ValueError("Every source_id must contain one human_control and one mixed row.")

    mixed_pairs = source[source["kind"] == "mixed"].copy()
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=GROUP_TEST_SIZE,
        random_state=SELECTION_SEED,
    )
    _, selected_indices = next(
        splitter.split(mixed_pairs, groups=mixed_pairs["group_id"])
    )
    selected_pairs = mixed_pairs.iloc[selected_indices]
    if len(selected_pairs) != EXPECTED_PAIRS:
        raise ValueError(
            f"Expected {EXPECTED_PAIRS} selected pairs, found {len(selected_pairs)}."
        )

    selected_source_ids = set(selected_pairs["source_id"])
    selected_group_ids = set(selected_pairs["group_id"])
    if set(source.loc[source["group_id"].isin(selected_group_ids), "source_id"]) != selected_source_ids:
        raise ValueError("A source group was split between training and challenge.")

    selected = source[source["source_id"].isin(selected_source_ids)].copy()
    selected["Class"] = selected["kind"].map({"mixed": 0, "human_control": 1})
    selected["Text"] = selected["text"].map(normalize_text)
    selected["kind_order"] = selected["kind"].map({"mixed": 0, "human_control": 1})
    selected = selected.sort_values(["source_id", "kind_order"])
    challenge_columns = [
        "Text", "Class", "source_id", "group_id", "source_dataset",
        "generator_backend", "ai_fraction_chars", "kind",
    ]
    challenge = selected[challenge_columns].reset_index(drop=True)
    if challenge["Class"].value_counts().to_dict() != {0: EXPECTED_PAIRS, 1: EXPECTED_PAIRS}:
        raise ValueError("Challenge set is not pair-balanced.")

    training = pd.read_csv(args.training_csv, sep=";", encoding="utf-8-sig")
    if len(training) != 12420:
        raise ValueError(f"Expected 12,420 original training rows, found {len(training)}.")
    prefix = training.iloc[:ORIGINAL_AND_HC3_ROWS].copy()
    heterogeneous = training.iloc[ORIGINAL_AND_HC3_ROWS:].copy()
    parquet_keys = set(source["text"].map(normalize_text))
    heterogeneous_keys = heterogeneous["Text"].map(normalize_text)
    if len(heterogeneous) != len(source) or set(heterogeneous_keys) != parquet_keys:
        raise ValueError("The training CSV Heterogeneous section does not match the Parquet file.")

    selected_keys = set(challenge["Text"])
    retained_heterogeneous = heterogeneous[~heterogeneous_keys.isin(selected_keys)].copy()
    revised_training = pd.concat([prefix, retained_heterogeneous], ignore_index=True)
    if len(revised_training) != len(training) - len(challenge):
        raise ValueError("Unexpected number of rows after removing the challenge set.")

    args.training_csv.parent.mkdir(parents=True, exist_ok=True)
    args.challenge_csv.parent.mkdir(parents=True, exist_ok=True)
    revised_training.to_csv(args.training_csv, sep=";", index=False, encoding="utf-8-sig")
    challenge.to_csv(args.challenge_csv, sep=";", index=False, encoding="utf-8-sig")

    print(f"Training rows: {len(training)} -> {len(revised_training)}")
    print(f"Challenge rows: {len(challenge)} ({EXPECTED_PAIRS} matched pairs)")
    print(f"Held-out source groups: {len(selected_group_ids)}")
    print(challenge.groupby(["source_dataset", "Class"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
