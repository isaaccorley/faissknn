"""Regression test for GH #14: GPU search not stream-ordered off cuda:0.

``faiss.contrib.torch_utils`` syncs faiss's stream against
``torch.cuda.current_device()``, not the device the index actually lives
on. Selecting a non-default GPU the idiomatic way (``.to("cuda:N")``, no
device context) left faiss's search racing against torch's producing /
consuming kernels, silently corrupting results on any GPU other than
``cuda:0``. See the issue for the full root-cause writeup.

This reproduces the race from the issue's repro script: a query tensor is
still being produced by pending matmuls on torch's stream when
``FaissKNNClassifier.predict`` is called on a non-default device.
"""

import numpy as np
import pytest

from faissknn import FaissKNNClassifier  # before torch on macOS (see knn.py)

# isort: split
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least 2 CUDA GPUs to exercise a non-default device",
)


def _pending_query(q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Return a tensor equal to ``q`` but only once queued matmuls finish."""
    h = q
    for _ in range(8):
        h = h @ w
    return q + 0.0 * h[:, :1]


@pytest.mark.parametrize("dev", ["cuda:1"])
def test_gpu_predict_matches_cpu_off_default_device(dev: str) -> None:
    """predict() on a non-default GPU must match the CPU reference every time.

    Before the fix this flipped on essentially every call: faiss's search
    ran on the index's private cuda:1 stream while the query tensor was
    still being written by pending matmuls on torch's current (cuda:0)
    stream, and the result read back before faiss finished writing it.
    """
    torch.manual_seed(0)
    n_train, n_test, d, n_classes, k = 2000, 200, 256, 10, 5

    x_train = torch.randn(n_train, d)
    y_train = torch.randint(0, n_classes, (n_train,))
    x_test = torch.randn(n_test, d)

    ref = FaissKNNClassifier(n_neighbors=k, device="cpu").fit(x_train, y_train).predict(x_test)

    clf = FaissKNNClassifier(n_neighbors=k, device=dev).fit(x_train.to(dev), y_train)
    w = torch.randn(d, d, device=dev)

    for _ in range(20):
        pred = clf.predict(_pending_query(x_test.to(dev), w))
        np.testing.assert_array_equal(pred, ref)
