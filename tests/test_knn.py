from collections.abc import Sequence

import numpy as np

from faissknn import FaissKNNClassifier, FaissKNNMultilabelClassifier


def test_multiclass_knn(multiclass_dataset: Sequence[np.ndarray]):
    x_train, y_train, x_test, _ = multiclass_dataset
    knn = FaissKNNClassifier(n_neighbors=5, n_classes=None, device="cpu")
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    y_proba = knn.predict_proba(x_test)
    assert y_pred.ndim == 1
    assert y_proba.ndim == 2


def test_multilabel_knn(multilabel_dataset: Sequence[np.ndarray]):
    x_train, y_train, x_test, _ = multilabel_dataset
    knn = FaissKNNMultilabelClassifier(n_neighbors=5, n_classes=None, device="cpu")
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    y_proba = knn.predict_proba(x_test)
    assert y_pred.ndim == 2
    assert y_proba.ndim == 2


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
    _, idx = knn.index.search(np.atleast_2d(x_test).astype(np.float32), k=7)
    class_idx = knn.y[idx]
    expected = _reference_multiclass(class_idx, knn.n_classes)
    np.testing.assert_array_equal(knn._class_counts(class_idx), expected)


def test_multilabel_predict_matches_reference(multilabel_dataset):
    """Vectorized sum-over-k must match the original per-label bincount loop."""
    x_train, y_train, x_test, _ = multilabel_dataset
    knn = FaissKNNMultilabelClassifier(n_neighbors=5, device="cpu")
    knn.fit(x_train, y_train)
    new = knn.predict(x_test)
    new_proba = knn.predict_proba(x_test)

    # Reproduce the old per-label loop locally for comparison.
    _, idx = knn.index.search(np.atleast_2d(x_test).astype(np.float32), k=5)
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
