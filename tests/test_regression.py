"""End-to-end regression tests against trusted reference implementations.

These tests ensure that public ``predict`` / ``predict_proba`` outputs from
``FaissKNNClassifier`` and ``FaissKNNMultilabelClassifier`` agree with
independent reference implementations on deterministic dummy data — so
refactors can't silently change semantics.

References:
- multiclass: ``sklearn.neighbors.KNeighborsClassifier`` (brute-force, identical
  metric); both rank by ascending squared / euclidean distance so the
  resulting neighbor sets and majority votes must match.
- multilabel: a small NumPy brute-force implementation in this file; sklearn
  doesn't ship a multilabel KNN that mirrors our voting rule exactly.
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier

from faissknn import FaissKNNClassifier, FaissKNNMultilabelClassifier


def _make_multiclass(n_train: int = 100, n_test: int = 20, d: int = 16, c: int = 5):
    """Deterministic multiclass dummy dataset."""
    rng = np.random.default_rng(42)
    x_train = rng.standard_normal((n_train, d)).astype(np.float32)
    y_train = rng.integers(0, c, size=n_train).astype(int)
    x_test = rng.standard_normal((n_test, d)).astype(np.float32)
    return x_train, y_train, x_test


def _make_multilabel(n_train: int = 100, n_test: int = 20, d: int = 16, L: int = 4):
    """Deterministic multilabel dummy dataset."""
    rng = np.random.default_rng(43)
    x_train = rng.standard_normal((n_train, d)).astype(np.float32)
    y_train = rng.integers(0, 2, size=(n_train, L)).astype(int)
    x_test = rng.standard_normal((n_test, d)).astype(np.float32)
    return x_train, y_train, x_test


def _brute_force_multilabel(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, k: int
):
    """Reference multilabel KNN: euclidean distance + per-label majority vote.

    Ties on per-label vote go to class 0 (matches argmax(bincount([z, ones]))
    behavior used internally by ``FaissKNNMultilabelClassifier``).
    """
    dists = cdist(x_test, x_train)
    nn = np.argsort(dists, axis=1, kind="stable")[:, :k]
    nn_labels = y_train[nn]  # (n_test, k, L)
    ones = nn_labels.sum(axis=1)  # (n_test, L)
    pred = (ones > k - ones).astype(int)
    proba = ones / k
    return pred, proba


@pytest.mark.parametrize("k", [1, 5, 11])
def test_multiclass_matches_sklearn(k: int, device: str):
    """faissknn predict / predict_proba must match sklearn brute-force KNN."""
    x_train, y_train, x_test = _make_multiclass()

    ours = FaissKNNClassifier(n_neighbors=k, device=device).fit(x_train, y_train)
    ref = KNeighborsClassifier(
        n_neighbors=k, algorithm="brute", metric="euclidean"
    ).fit(x_train, y_train)

    np.testing.assert_array_equal(ours.predict(x_test), ref.predict(x_test))
    # sklearn returns probabilities sorted by ascending class label; our
    # bincount over y values does the same since y values *are* the class
    # indices, so columns line up.
    np.testing.assert_allclose(
        ours.predict_proba(x_test),
        ref.predict_proba(x_test),
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.parametrize("k", [1, 5, 11])
def test_multilabel_matches_brute_force(k: int, device: str):
    """Multilabel predict / predict_proba must match a brute-force NumPy ref."""
    x_train, y_train, x_test = _make_multilabel()

    ours = FaissKNNMultilabelClassifier(n_neighbors=k, device=device).fit(x_train, y_train)
    ref_pred, ref_proba = _brute_force_multilabel(x_train, y_train, x_test, k)

    np.testing.assert_array_equal(ours.predict(x_test), ref_pred)
    np.testing.assert_allclose(
        ours.predict_proba(x_test), ref_proba, rtol=0, atol=1e-6
    )
