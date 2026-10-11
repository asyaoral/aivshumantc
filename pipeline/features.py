"""Six deterministic linguistic features from the original project."""
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES = [
    "Avg_Sentence_Length",
    "Lexical_Diversity",
    "Punctuation_Density",
    "Capitalized_Ratio",
    "Pronoun_Ratio",
    "Num_Sentences",
]


class LinguisticFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            texts = X.tolist()
        else:
            texts = np.asarray(X).reshape(-1).tolist()

        rows = []
        pronouns = {"i", "we", "my", "our", "us", "their"}
        for value in texts:
            text = str(value)
            words = text.split()
            word_count = max(1, len(words))
            sentences = [item.strip() for item in re.split(r"[.!?]", text) if item.strip()]
            sentence_count = len(sentences)
            rows.append([
                len(words) / sentence_count if sentence_count else 0.0,
                len(set(words)) / word_count,
                sum(character in ".,;:!?" for character in text) / word_count,
                sum(len(word) > 1 and word[0].isupper() for word in words) / word_count,
                sum(word.lower() in pronouns for word in words) / word_count,
                float(sentence_count),
            ])
        return np.asarray(rows, dtype=float)

