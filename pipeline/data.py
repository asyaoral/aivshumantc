"""Dataset loading, cleaning, leakage checks, and pair grouping."""
from pathlib import Path
from itertools import combinations
import re

import pandas as pd

from .config import HC3_END_ROW, MIN_WORDS, ORIGINAL_TRAINING_ROWS


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as stream:
        header = stream.readline()
    separator = ";" if ";" in header and header.count(";") >= header.count(",") else ","
    frame = pd.read_csv(path, sep=separator, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    normalized = {column.lower().strip(): column for column in frame.columns}
    if "text" not in normalized:
        raise ValueError(f"Expected a text column in {path}; found: {list(frame.columns)}")
    label = normalized.get("class") or normalized.get("generated")
    if label is None:
        raise ValueError(f"Expected Class or generated column in {path}; found: {list(frame.columns)}")
    return frame.rename(columns={normalized["text"]: "Text", label: "Class"})


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_dataset(raw: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, dict]:
    work = raw[["Text", "Class"]].copy()
    work["source_row"] = work.index.astype(int)
    work = work.dropna(subset=["Text", "Class"])
    work["Text"] = work["Text"].map(normalize_text)
    work = work[work["Text"].str.len() > 0].copy()

    if pd.api.types.is_numeric_dtype(work["Class"]):
        work["Class"] = pd.to_numeric(work["Class"], errors="coerce")
    else:
        labels = {
            "0": 0, "1": 1, "ai": 0, "ai-generated": 0, "ai generated": 0,
            "human": 1, "human-written": 1, "human written": 1,
        }
        work["Class"] = work["Class"].astype(str).str.strip().str.lower().map(labels)
    if work["Class"].isna().any() or not work["Class"].isin([0, 1]).all():
        raise ValueError(f"Unexpected labels in {dataset_name}. Expected 0/1 or AI/Human.")
    work["Class"] = work["Class"].astype(int)

    work["word_count"] = work["Text"].str.split().str.len()
    short_removed = int((work["word_count"] < MIN_WORDS).sum())
    work = work[work["word_count"] >= MIN_WORDS].copy()

    work["text_key"] = work["Text"].str.casefold()
    conflict_counts = work.groupby("text_key")["Class"].nunique()
    conflicting_keys = set(conflict_counts[conflict_counts > 1].index)
    conflicting_rows = int(work["text_key"].isin(conflicting_keys).sum())
    work = work[~work["text_key"].isin(conflicting_keys)].copy()
    before_deduplication = len(work)
    work = work.drop_duplicates(subset=["text_key"], keep="first")

    cleaned = work[["Text", "Class", "word_count", "text_key", "source_row"]].reset_index(drop=True)
    report = {
        "Dataset": dataset_name,
        "Original rows": len(raw),
        "Short texts excluded": short_removed,
        "Conflicting-label text groups": len(conflicting_keys),
        "Conflicting-label rows excluded": conflicting_rows,
        "Same-label duplicates excluded": before_deduplication - len(work),
        "Usable unique texts": len(cleaned),
        "AI (0)": int((cleaned["Class"] == 0).sum()),
        "Human (1)": int((cleaned["Class"] == 1).sum()),
    }
    return cleaned, report


def verify_no_overlap(**datasets: pd.DataFrame) -> None:
    """Reject exact normalized text overlap across any named datasets."""
    overlaps = {}
    for (left_name, left), (right_name, right) in combinations(datasets.items(), 2):
        overlap = set(left["text_key"]) & set(right["text_key"])
        if overlap:
            overlaps[f"{left_name}/{right_name}"] = len(overlap)
    if overlaps:
        details = ", ".join(f"{pair}={count}" for pair, count in overlaps.items())
        raise ValueError(f"Data leakage detected across datasets: {details}.")


def training_group(source_row: int) -> str:
    if ORIGINAL_TRAINING_ROWS <= source_row < HC3_END_ROW:
        return f"hc3_pair_{(source_row - ORIGINAL_TRAINING_ROWS) // 2}"
    if source_row >= HC3_END_ROW:
        return f"heterogeneous_pair_{(source_row - HC3_END_ROW) // 2}"
    return f"original_{source_row}"

