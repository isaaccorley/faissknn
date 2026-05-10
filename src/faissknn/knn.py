"""FAISS-based KNN classifiers for multiclass and multilabel classification."""

from typing import Literal

import faiss
import numpy as np

Metric = Literal["l2", "ip", "cosine"]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize; safe against zero rows."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


class FaissKNNClassifier:
    """
    A multiclass exact KNN classifier implemented using the FAISS library.

    Parameters
    ----------
    n_neighbors : int
        Number of KNN neighbors
    n_classes : int, optional
        Number of dataset classes (otherwise derived from the data)
    device : str, default="cpu"
        A torch device, e.g. cpu, cuda, cuda:0, etc.
    metric : {"l2", "ip", "cosine"}, default="l2"
        Distance metric used to rank neighbors:
          - "l2"     : squared Euclidean (FAISS IndexFlatL2)
          - "ip"     : inner product (FAISS IndexFlatIP); useful when vectors
                       are already normalized or when raw dot-product ranking
                       is desired
          - "cosine" : cosine similarity; equivalent to "ip" but the
                       classifier L2-normalizes the inputs on `fit` and
                       `predict` so the user does not have to
    """

    def __init__(
        self,
        n_neighbors: int,
        n_classes: int | None = None,
        device: str = "cpu",
        metric: Metric = "l2",
    ) -> None:
        """Instantiate a faiss KNN Classifier."""
        if metric not in ("l2", "ip", "cosine"):
            msg = f"metric must be one of 'l2', 'ip', 'cosine'; got {metric!r}"
            raise ValueError(msg)
        self.n_neighbors = n_neighbors
        self.n_classes = n_classes
        self.metric = metric

        if device == "cpu":
            self.cuda = False
            self.device = None
        else:
            self.cuda = True
            if ":" in device:
                self.device = int(device.split(":")[-1])
            else:
                self.device = 0

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        """Apply dtype + contiguity + metric-specific preprocessing."""
        x = np.atleast_2d(x).astype(np.float32)
        x = np.ascontiguousarray(x)
        if self.metric == "cosine":
            x = _l2_normalize(x)
        return x

    def create_index(self, d: int) -> None:
        """Create the faiss index."""
        use_ip = self.metric in ("ip", "cosine")
        if self.cuda:
            self.res = faiss.StandardGpuResources()  # type: ignore[possibly-missing-attribute]
            if use_ip:
                self.config = faiss.GpuIndexFlatConfig()
                self.config.device = self.device
                self.index = faiss.GpuIndexFlatIP(self.res, d, self.config)
            else:
                self.config = faiss.GpuIndexFlatConfig()
                self.config.device = self.device
                self.index = faiss.GpuIndexFlatL2(self.res, d, self.config)
        else:
            self.index = faiss.IndexFlatIP(d) if use_ip else faiss.IndexFlatL2(d)

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        """Store train X and y."""
        X = self._prepare(X)
        self.create_index(X.shape[-1])
        self.index.add(X)  # type: ignore[arg-type]
        self.y = y.astype(int)
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))
        return self

    def __del__(self) -> None:
        """Cleanup helpers."""
        if hasattr(self, "index"):
            self.index.reset()
            del self.index
        if hasattr(self, "res"):
            self.res.noTempMemory()
            del self.res

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict int labels given X."""
        X = self._prepare(X)
        _, idx = self.index.search(X, k=self.n_neighbors)  # type: ignore[missing-argument]
        class_idx = self.y[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes),  # type: ignore[invalid-argument-type]
            axis=1,
            arr=class_idx.astype(np.int16),
        )
        return np.argmax(counts, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict float probabilities for labels given X."""
        X = self._prepare(X)
        _, idx = self.index.search(X, k=self.n_neighbors)  # type: ignore[missing-argument]
        class_idx = self.y[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes),  # type: ignore[invalid-argument-type]
            axis=1,
            arr=class_idx.astype(np.int16),
        )
        return counts / self.n_neighbors


class FaissKNNMultilabelClassifier(FaissKNNClassifier):
    """A multilabel exact KNN classifier implemented using the FAISS library."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict one-hot int labels given X."""
        X = self._prepare(X)
        _, idx = self.index.search(X, k=self.n_neighbors)  # type: ignore[missing-argument]
        class_idx = self.y[idx]

        preds = []
        for i in range(class_idx.shape[-1]):
            class_idx_i = class_idx[..., i]
            counts_i = np.apply_along_axis(
                lambda x: np.bincount(x, minlength=2),
                axis=1,
                arr=class_idx_i.astype(np.int16),
            )
            preds_i = np.argmax(counts_i, axis=1)
            preds.append(preds_i)

        return np.stack(preds, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict float probabilities for labels given X."""
        X = self._prepare(X)
        _, idx = self.index.search(X, k=self.n_neighbors)  # type: ignore[missing-argument]
        class_idx = self.y[idx]

        preds_proba = []
        for i in range(class_idx.shape[-1]):
            class_idx_i = class_idx[..., i]
            counts_i = np.apply_along_axis(
                lambda x: np.bincount(x, minlength=2),
                axis=1,
                arr=class_idx_i.astype(np.int16),
            )
            preds_proba_i = counts_i / self.n_neighbors
            preds_proba_i = preds_proba_i[:, 1]
            preds_proba.append(preds_proba_i)

        return np.stack(preds_proba, axis=1)
