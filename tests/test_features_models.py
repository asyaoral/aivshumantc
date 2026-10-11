from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features import FEATURE_NAMES, LinguisticFeatures
from pipeline.models import make_model


def test_linguistic_features_have_expected_shape_and_finite_values():
    texts = pd.Series([
        "I write one sentence. We write another sentence!",
        "A compact example has punctuation, capitals, and words.",
    ])
    values = LinguisticFeatures().fit_transform(texts)
    assert values.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(values).all()


@pytest.mark.parametrize("name", ["Naive Bayes", "Logistic Regression", "SVM"])
def test_supported_models_build_valid_pipelines(name):
    model = make_model(name)
    assert list(model.named_steps) == ["features", "classifier"]


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("Unsupported")


def test_multinomial_naive_bayes_rejects_scaled_manual_features():
    with pytest.raises(ValueError, match="non-negative"):
        make_model("Naive Bayes", add_manual_features=True)
