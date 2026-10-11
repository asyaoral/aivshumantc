"""Run pytest -q before publishing to GitHub."""
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from src.preprocess import clean_dataset
from src.models import build_pipeline, candidate_configs


def test_conflicting_labels_and_duplicates_are_removed():
    long = "One two three four five six "
    df = pd.DataFrame({
        "Text": [long, long, long.replace("One", "Another"),
                 long.replace("One", "Another"), "short"],
        "Class": [0, 1, 0, 0, 1],
    })
    clean, r = clean_dataset(df, min_words=5)
    assert len(clean) == 1
    assert r["conflicting_text_groups"] == 1
    assert r["duplicate_rows_removed"] == 1
    assert r["too_short_removed"] == 1


def test_wrong_label_rejected():
    with pytest.raises(ValueError, match="Unexpected labels"):
        clean_dataset(pd.DataFrame({"Text": ["long enough for a test"], "Class": ["unknown"]}), 3)


def test_all_candidate_pipelines_predict():
    texts = ["Human writing about parks with sentences and examples " + str(i) for i in range(12)] + [
        "AI-generated description of computer systems and logical flow " + str(i) for i in range(12)
    ]
    labels = [1] * 12 + [0] * 12
    df = pd.DataFrame({"Text": texts})
    xtr, xte, ytr, _ = train_test_split(df, labels, test_size=0.25, stratify=labels, random_state=42)
    for conf in candidate_configs():
        m = build_pipeline(conf["model"], remove_stop_words=conf["remove_stop_words"],
                           manual_features=conf["manual_features"])
        m.fit(xtr, ytr)
        assert len(m.predict(xte)) == len(xte)
