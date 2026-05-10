from collections.abc import Sequence

import numpy as np

from faissknn import FaissKNNClassifier, FaissKNNMultilabelClassifier


def test_multiclass_knn(device: str, multiclass_dataset: Sequence[np.ndarray]):
    x_train, y_train, x_test, _ = multiclass_dataset
    knn = FaissKNNClassifier(n_neighbors=5, n_classes=None, device=device)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    y_proba = knn.predict_proba(x_test)
    assert y_pred.ndim == 1
    assert y_proba.ndim == 2


def test_multilabel_knn(device: str, multilabel_dataset: Sequence[np.ndarray]):
    x_train, y_train, x_test, _ = multilabel_dataset
    knn = FaissKNNMultilabelClassifier(n_neighbors=5, n_classes=None, device=device)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    y_proba = knn.predict_proba(x_test)
    assert y_pred.ndim == 2
    assert y_proba.ndim == 2


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
