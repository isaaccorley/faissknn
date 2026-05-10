from collections.abc import Sequence

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.model_selection import train_test_split


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--device {cpu,cuda}`` to switch the test fixture device.

    Default is ``cpu``. With ``--device=cuda`` the ``device`` fixture
    yields ``"cuda"`` (skipping the test if torch/cuda isn't available).
    """
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device to run faissknn tests on.",
    )


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest) -> str:
    """Return the test device; skip if cuda was requested but unavailable."""
    d = request.config.getoption("--device")
    if d == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("CUDA requested but torch.cuda.is_available() is False")
        except ImportError:
            pytest.skip("CUDA requested but torch is not installed")
    return d


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
