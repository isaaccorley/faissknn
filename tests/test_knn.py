from collections.abc import Sequence

import numpy as np
import pytest

from faissknn import FaissKNNClassifier, FaissKNNMultilabelClassifier


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_multiclass_knn(
    metric: str, device: str, multiclass_dataset: Sequence[np.ndarray]
):
    x_train, y_train, x_test, _ = multiclass_dataset
    knn = FaissKNNClassifier(n_neighbors=5, n_classes=None, device=device, metric=metric)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    y_proba = knn.predict_proba(x_test)
    assert y_pred.ndim == 1
    assert y_proba.ndim == 2


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_multilabel_knn(
    metric: str, device: str, multilabel_dataset: Sequence[np.ndarray]
):
    x_train, y_train, x_test, _ = multilabel_dataset
    knn = FaissKNNMultilabelClassifier(
        n_neighbors=5, n_classes=None, device=device, metric=metric
    )
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    y_proba = knn.predict_proba(x_test)
    assert y_pred.ndim == 2
    assert y_proba.ndim == 2


def test_invalid_metric_raises():
    with pytest.raises(ValueError, match="metric must be one of"):
        FaissKNNClassifier(n_neighbors=5, metric="manhattan")  # type: ignore[arg-type]


def test_cosine_normalizes_inputs(device: str):
    """cosine should L2-normalize so vector magnitude doesn't change predictions."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 8)).astype(np.float32)
    y = rng.integers(0, 3, size=50)
    query = rng.standard_normal((5, 8)).astype(np.float32)

    knn1 = FaissKNNClassifier(n_neighbors=3, metric="cosine", device=device).fit(x, y)
    knn2 = FaissKNNClassifier(n_neighbors=3, metric="cosine", device=device).fit(x * 100.0, y)
    np.testing.assert_array_equal(knn1.predict(query), knn2.predict(query * 0.01))
