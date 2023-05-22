from typing import Sequence

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="package")
def multilabel_dataset() -> Sequence[np.ndarray]:
    seed = 0
    x, y = make_multilabel_classification(
        n_samples=200,
        n_features=32,
        n_classes=10,
        n_labels=2,
        allow_unlabeled=False,
        sparse=False,
        random_state=seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=seed
    )
    return x_train, y_train, x_test, y_test


@pytest.fixture(scope="package")
def multiclass_dataset() -> Sequence[np.ndarray]:
    seed = 0
    x, y = make_classification(
        n_samples=200, n_features=32, n_classes=10, n_informative=10, random_state=seed
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=seed
    )
    return x_train, y_train, x_test, y_test
