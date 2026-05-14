from collections.abc import Sequence

import numpy as np
import pytest

from faissknn import FaissKNNClassifier, FaissKNNMultilabelClassifier


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_multiclass_knn(metric: str, device: str, multiclass_dataset: Sequence[np.ndarray]):
    x_train, y_train, x_test, _ = multiclass_dataset
    knn = FaissKNNClassifier(n_neighbors=5, n_classes=None, device=device, metric=metric)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    y_proba = knn.predict_proba(x_test)
    assert y_pred.ndim == 1
    assert y_proba.ndim == 2


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_multilabel_knn(metric: str, device: str, multilabel_dataset: Sequence[np.ndarray]):
    x_train, y_train, x_test, _ = multilabel_dataset
    knn = FaissKNNMultilabelClassifier(n_neighbors=5, n_classes=None, device=device, metric=metric)
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


def test_use_fp16_is_cpu_safe(multiclass_dataset):
    """use_fp16 is a GPU-only knob — must be a silent no-op on CPU."""
    x_train, y_train, x_test, _ = multiclass_dataset
    knn = FaissKNNClassifier(n_neighbors=5, device="cpu", use_fp16=True)
    knn.fit(x_train, y_train)
    assert knn.predict(x_test).shape == (len(x_test),)


def test_use_fp16_on_device(device: str, multiclass_dataset):
    """use_fp16=True should produce sensible predictions on the configured device."""
    x_train, y_train, x_test, _ = multiclass_dataset
    knn = FaissKNNClassifier(n_neighbors=5, device=device, use_fp16=True)
    knn.fit(x_train, y_train)
    assert knn.predict(x_test).shape == (len(x_test),)


def test_accepts_torch_tensors(device: str, multiclass_dataset):
    """fit/predict should accept torch tensors and produce identical output."""
    import torch

    x_train, y_train, x_test, _ = multiclass_dataset
    np_knn = FaissKNNClassifier(n_neighbors=5, device=device)
    np_knn.fit(x_train, y_train)
    np_pred = np_knn.predict(x_test)

    t_knn = FaissKNNClassifier(n_neighbors=5, device=device)
    t_knn.fit(torch.from_numpy(x_train), torch.from_numpy(y_train))
    t_pred = t_knn.predict(torch.from_numpy(x_test))

    np.testing.assert_array_equal(np_pred, t_pred)


def _reference_multiclass(class_idx: np.ndarray, n_classes: int) -> np.ndarray:
    """Original per-row bincount loop — kept here as the golden reference."""
    return np.apply_along_axis(
        lambda x: np.bincount(x, minlength=n_classes),
        axis=1,
        arr=class_idx.astype(np.int16),
    )


def test_multiclass_predict_matches_reference(multiclass_dataset):
    """Vectorized scatter-add output must match the original bincount loop."""
    x_train, y_train, x_test, _ = multiclass_dataset
    knn = FaissKNNClassifier(n_neighbors=7, device="cpu")
    knn.fit(x_train, y_train)
    # faiss.contrib.torch_utils monkey-patches search to a pythonic (x, k)
    # signature at import time; the static SWIG signature still shows the
    # raw (n, x, k, distances, labels) form.
    _, idx = knn.index.search(np.atleast_2d(x_test).astype(np.float32), k=7)  # ty: ignore[missing-argument]
    class_idx = knn.y[idx]
    assert knn.n_classes is not None  # set during fit
    expected = _reference_multiclass(class_idx, knn.n_classes)
    np.testing.assert_array_equal(knn._class_counts(class_idx), expected)


def test_multilabel_predict_matches_reference(multilabel_dataset):
    """Vectorized sum-over-k must match the original per-label bincount loop."""
    x_train, y_train, x_test, _ = multilabel_dataset
    knn = FaissKNNMultilabelClassifier(n_neighbors=5, device="cpu")
    knn.fit(x_train, y_train)
    new = knn.predict(x_test)
    new_proba = knn.predict_proba(x_test)

    _, idx = knn.index.search(np.atleast_2d(x_test).astype(np.float32), k=5)  # ty: ignore[missing-argument]
    class_idx = knn.y[idx]
    ref_preds = []
    ref_probas = []
    for i in range(class_idx.shape[-1]):
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=2),
            axis=1,
            arr=class_idx[..., i].astype(np.int16),
        )
        ref_preds.append(np.argmax(counts, axis=1))
        ref_probas.append(counts[:, 1] / 5)
    np.testing.assert_array_equal(new, np.stack(ref_preds, axis=1))
    np.testing.assert_allclose(new_proba, np.stack(ref_probas, axis=1))
