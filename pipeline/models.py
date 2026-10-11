"""Leakage-safe model construction and training-CV comparison."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import CV_FOLDS, MAX_FEATURES, RANDOM_STATE
from .features import LinguisticFeatures


def make_model(name: str, *, remove_stop_words: bool = False,
               add_manual_features: bool = False) -> Pipeline:
    models = {
        "Naive Bayes": lambda: MultinomialNB(),
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "SVM": lambda: LinearSVC(max_iter=10000, random_state=RANDOM_STATE),
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}")
    if name == "Naive Bayes" and add_manual_features:
        raise ValueError(
            "Multinomial Naive Bayes requires non-negative inputs; "
            "the scaled manual features may contain negative values."
        )

    feature_steps = [(
        "tfidf",
        TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=(1, 2),
            stop_words="english" if remove_stop_words else None,
        ),
        "Text",
    )]
    if add_manual_features:
        feature_steps.append((
            "manual",
            Pipeline([
                ("extract", LinguisticFeatures()),
                ("scale", StandardScaler()),
            ]),
            "Text",
        ))
    return Pipeline([
        ("features", ColumnTransformer(feature_steps)),
        ("classifier", models[name]()),
    ])


def evaluate_candidates(X_train: pd.DataFrame, y_train: pd.Series,
                        training_groups: pd.Series) -> pd.DataFrame:
    cv = StratifiedGroupKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    experiments = [
        ("Naive Bayes | TF-IDF (keep stop words)", "Naive Bayes", False, False),
        ("Logistic Regression | TF-IDF (keep stop words)", "Logistic Regression", False, False),
        ("Logistic Regression | TF-IDF (remove stop words)", "Logistic Regression", True, False),
        ("Logistic Regression | TF-IDF + 6 linguistic features", "Logistic Regression", False, True),
        ("SVM | TF-IDF (keep stop words)", "SVM", False, False),
        ("SVM | TF-IDF (remove stop words)", "SVM", True, False),
        ("SVM | TF-IDF + 6 linguistic features", "SVM", False, True),
    ]
    rows = []
    for title, model_name, remove_stop_words, manual_features in experiments:
        scores = cross_validate(
            make_model(
                model_name,
                remove_stop_words=remove_stop_words,
                add_manual_features=manual_features,
            ),
            X_train,
            y_train,
            cv=cv,
            groups=training_groups,
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
            n_jobs=1,
            error_score="raise",
        )
        rows.append({
            "Experiment": title,
            "Model": model_name,
            "Remove stop words": remove_stop_words,
            "Manual features": manual_features,
            "CV Accuracy": np.mean(scores["test_accuracy"]),
            "CV F1 (macro)": np.mean(scores["test_f1_macro"]),
            "CV F1 std": np.std(scores["test_f1_macro"]),
        })
    return pd.DataFrame(rows).sort_values("CV F1 (macro)", ascending=False).reset_index(drop=True)

