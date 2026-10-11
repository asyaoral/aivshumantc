"""Six hand-crafted language-style features from the team's earlier ML project."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES = [
    "Avg_Sentence_Length", "Lexical_Diversity", "Punctuation_Density",
    "Capitalized_Ratio", "Pronoun_Ratio", "Num_Sentences",
]


class LinguisticFeatures(BaseEstimator, TransformerMixin):
    """Stateless sklearn transformer. Tokenization is whitespace-based for reproducibility."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            texts = X.tolist()
        else:
            texts = np.asarray(X).reshape(-1).tolist()
        pronouns = {"i", "we", "my", "our", "us", "their"}
        rows = []
        for value in texts:
            text = str(value)
            words = text.split()
            n = max(len(words), 1)
            sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
            ns = len(sentences)
            rows.append([
                len(words) / ns if ns else 0.0,
                len(set(words)) / n,
                sum(c in ".,;:!?" for c in text) / n,
                sum(len(w) > 1 and w[0].isupper() for w in words) / n,
                sum(w.lower() in pronouns for w in words) / n,
                float(ns),
            ])
        return np.asarray(rows, dtype=float)
