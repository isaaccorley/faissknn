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


def test_accepts_torch_tensors(multiclass_dataset):
    """fit/predict should accept torch tensors and produce identical output."""
    import torch

    x_train, y_train, x_test, _ = multiclass_dataset
    np_knn = FaissKNNClassifier(n_neighbors=5, device="cpu")
    np_knn.fit(x_train, y_train)
    np_pred = np_knn.predict(x_test)

    t_knn = FaissKNNClassifier(n_neighbors=5, device="cpu")
    t_knn.fit(torch.from_numpy(x_train), torch.from_numpy(y_train))
    t_pred = t_knn.predict(torch.from_numpy(x_test))

    np.testing.assert_array_equal(np_pred, t_pred)
