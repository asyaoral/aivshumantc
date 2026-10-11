from __future__ import annotations

import pandas as pd
import pytest

from pipeline.config import HC3_END_ROW, MIN_WORDS, ORIGINAL_TRAINING_ROWS
from pipeline.data import clean_dataset, load_dataset, training_group, verify_no_overlap


def words(token: str, count: int = MIN_WORDS) -> str:
    return " ".join([token] * count)


def test_load_dataset_detects_semicolon_and_standardizes_columns(tmp_path):
    source = tmp_path / "sample.csv"
    source.write_text(
        f"text;generated\n{words('human')};1\n",
        encoding="utf-8-sig",
    )
    loaded = load_dataset(source)
    assert list(loaded.columns) == ["Text", "Class"]
    assert loaded.iloc[0]["Class"] == 1


def test_clean_dataset_filters_short_conflicts_and_duplicates():
    duplicate_ai = words("machine")
    unique_human = words("person")
    conflicting = words("conflict")
    raw = pd.DataFrame({
        "Text": [
            duplicate_ai,
            f"  {duplicate_ai}  ",
            unique_human,
            conflicting,
            conflicting,
            "too short",
        ],
        "Class": ["AI", "0", "Human", "AI", "Human", "Human"],
    })
    cleaned, report = clean_dataset(raw, "unit-test")
    assert len(cleaned) == 2
    assert set(cleaned["Class"]) == {0, 1}
    assert report["Short texts excluded"] == 1
    assert report["Conflicting-label text groups"] == 1
    assert report["Conflicting-label rows excluded"] == 2
    assert report["Same-label duplicates excluded"] == 1


def test_clean_dataset_rejects_unknown_labels():
    raw = pd.DataFrame({"Text": [words("sample")], "Class": ["unknown"]})
    with pytest.raises(ValueError, match="Unexpected labels"):
        clean_dataset(raw, "unit-test")


def test_verify_no_overlap_rejects_exact_normalized_leakage():
    left = pd.DataFrame({"text_key": ["shared text", "left only"]})
    right = pd.DataFrame({"text_key": ["shared text", "right only"]})
    with pytest.raises(ValueError, match="Data leakage detected"):
        verify_no_overlap(training=left, external=right)


def test_training_group_keeps_appended_pairs_together():
    assert training_group(ORIGINAL_TRAINING_ROWS - 1).startswith("original_")
    assert training_group(ORIGINAL_TRAINING_ROWS) == training_group(ORIGINAL_TRAINING_ROWS + 1)
    assert training_group(HC3_END_ROW) == training_group(HC3_END_ROW + 1)
    assert training_group(ORIGINAL_TRAINING_ROWS) != training_group(ORIGINAL_TRAINING_ROWS + 2)
