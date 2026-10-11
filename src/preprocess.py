"""Load, validate and clean the Text/Class CSV. Usage: python -m src.preprocess --input data/raw/main.csv"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import pandas as pd

LABEL_MAP = {
    "0": 0, "1": 1, "ai": 0, "ai-generated": 0, "ai generated": 0,
    "human": 1, "human-written": 1, "human written": 1,
}


def normalize_text(value: object) -> str:
    """Preserve punctuation and stop words (useful for authorship style)."""
    return re.sub(r"\s+", " ", str(value)).strip()


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Accept Apple Numbers CSV (comma or semicolon separated), with Text and Class columns."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected .csv, received {path.suffix}. Export .numbers to CSV using Apple Numbers > File > Export To > CSV.")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        first_line = stream.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    colmap = {c.casefold(): c for c in df.columns}
    if "text" not in colmap or "class" not in colmap:
        raise ValueError(f"Need Text and Class columns, found {df.columns.tolist()}. Export the relevant Numbers table as CSV.")
    return df.rename(columns={colmap["text"]: "Text", colmap["class"]: "Class"})[["Text", "Class"]]


def clean_dataset(raw: pd.DataFrame, min_words: int = 50) -> tuple[pd.DataFrame, dict]:
    """Deduplicate and remove contradictory-label groups before splitting.

    Texts shorter than min_words are excluded. Near duplicates are NOT detected.
    """
    if not {"Text", "Class"}.issubset(raw):
        raise ValueError("DataFrame must contain Text and Class.")
    work = raw[["Text", "Class"]].dropna().copy()
    work["Text"] = work["Text"].map(normalize_text)
    work = work.loc[work["Text"].ne("")].copy()
    empty_count = len(raw) - len(work)

    # Check labels; unlike an implicit cast, this does not silently accept arbitrary numbers.
    mapped = work["Class"].astype(str).str.strip().str.casefold().map(LABEL_MAP)
    if mapped.isna().any():
        bad = work.loc[mapped.isna(), "Class"].astype(str).unique()[:5]
        raise ValueError(f"Unexpected labels: {list(bad)}; expected 0/1 or AI/Human.")
    work["Class"] = mapped.astype(int)
    work["word_count"] = work["Text"].str.split().str.len()
    too_short = int((work["word_count"] < min_words).sum())
    work = work.loc[work["word_count"] >= min_words].copy()

    work["text_key"] = work["Text"].str.casefold()
    counts = work.groupby("text_key", sort=False)["Class"].nunique()
    conflicts = set(counts[counts > 1].index)
    conflicting_rows = int(work["text_key"].isin(conflicts).sum())
    work = work.loc[~work["text_key"].isin(conflicts)].copy()
    before_unique = len(work)
    work = work.drop_duplicates(subset=["text_key"], keep="first")
    duplicates = before_unique - len(work)

    clean = work[["Text", "Class"]].reset_index(drop=True)
    report = {
        "raw_rows": int(len(raw)),
        "blank_or_missing_removed": int(empty_count),
        "too_short_removed": int(too_short),
        "conflicting_text_groups": int(len(conflicts)),
        "conflicting_rows_removed": int(conflicting_rows),
        "duplicate_rows_removed": int(duplicates),
        "clean_rows": int(len(clean)),
        "ai_count": int((clean["Class"] == 0).sum()),
        "human_count": int((clean["Class"] == 1).sum()),
        "minimum_words": int(min_words),
    }
    return clean, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible dataset cleaning (NO model fitting)")
    parser.add_argument("--input", required=True, help="CSV exported from Apple Numbers")
    parser.add_argument("--output", default="data/processed/cleaned.csv")
    parser.add_argument("--report", default="outputs/data_quality.json")
    parser.add_argument("--min-words", type=int, default=50)
    args = parser.parse_args()
    clean, report = clean_dataset(read_dataset(args.input), min_words=args.min_words)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved cleaned data → {args.output}; report → {args.report}")


if __name__ == "__main__":
    main()
