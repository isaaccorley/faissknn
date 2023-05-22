from typing import Sequence

import numpy as np

from faissknn import FaissKNNClassifier, FaissKNNMultilabelClassifier


def test_multiclass_knn(multiclass_dataset: Sequence[np.ndarray]):
    x_train, y_train, x_test, y_test = multiclass_dataset
    knn = FaissKNNClassifier(n_neighbors=5, n_classes=None, device="cpu")
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    y_proba = knn.predict_proba(x_test)
    assert y_pred.ndim == 1
    assert y_proba.ndim == 2


def test_multilabel_knn(multilabel_dataset: Sequence[np.ndarray]):
    x_train, y_train, x_test, y_test = multilabel_dataset
    knn = FaissKNNMultilabelClassifier(n_neighbors=5, n_classes=None, device="cpu")
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    y_proba = knn.predict_proba(x_test)
    assert y_pred.ndim == 2
    assert y_proba.ndim == 2
