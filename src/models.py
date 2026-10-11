"""Sklearn pipelines with fold-local TF-IDF fitting (no CV vocabulary leakage)."""
from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .features import LinguisticFeatures

RANDOM_STATE = 42


def build_pipeline(model_name: str, *, remove_stop_words: bool = False,
                   manual_features: bool = False, max_features: int = 5000) -> Pipeline:
    estimators = {
        "Naive Bayes": lambda: MultinomialNB(),
        "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Linear SVM": lambda: LinearSVC(max_iter=10000, random_state=RANDOM_STATE),
    }
    if model_name not in estimators:
        raise ValueError(f"Unknown model {model_name!r}")
    if manual_features and model_name == "Naive Bayes":
        # StandardScaler can create negatives, violating MultinomialNB assumptions.
        raise ValueError("Linguistic feature ablation is designed for Linear SVM, not MultinomialNB")
    columns = [
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                 stop_words="english" if remove_stop_words else None), "Text")
    ]
    if manual_features:
        columns.append(("manual", Pipeline([
            ("linguistic", LinguisticFeatures()), ("scale", StandardScaler())
        ]), "Text"))
    return Pipeline([
        ("features", ColumnTransformer(columns)),
        ("classifier", estimators[model_name]()),
    ])


def candidate_configs() -> list[dict]:
    return [
        {"name": "Naive Bayes", "model": "Naive Bayes", "remove_stop_words": False, "manual_features": False},
        {"name": "Logistic Regression", "model": "Logistic Regression", "remove_stop_words": False, "manual_features": False},
        {"name": "Linear SVM", "model": "Linear SVM", "remove_stop_words": False, "manual_features": False},
        {"name": "Linear SVM / no stop words", "model": "Linear SVM", "remove_stop_words": True, "manual_features": False},
        {"name": "Linear SVM / 6 linguistic features", "model": "Linear SVM", "remove_stop_words": False, "manual_features": True},
    ]
