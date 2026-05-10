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
    """A multiclass exact KNN classifier implemented using the FAISS library.

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
          - "ip"     : inner product (FAISS IndexFlatIP)
          - "cosine" : cosine similarity; equivalent to "ip" but inputs are
                       L2-normalized on fit and predict so the user does not
                       have to
    use_fp16 : bool, default=False
        Run distance computation in fp16 on the GPU (~30% faster, half the
        index memory on Ampere+; effectively no recall loss for typical k).
        Vectors are still passed in as fp32. Ignored on the CPU index.
    """

    def __init__(
        self,
        n_neighbors: int,
        n_classes: int | None = None,
        device: str = "cpu",
        metric: Metric = "l2",
        use_fp16: bool = False,
    ) -> None:
        """Instantiate a faiss KNN Classifier."""
        if metric not in ("l2", "ip", "cosine"):
            msg = f"metric must be one of 'l2', 'ip', 'cosine'; got {metric!r}"
            raise ValueError(msg)
        self.n_neighbors = n_neighbors
        self.n_classes = n_classes
        self.metric = metric
        self.use_fp16 = use_fp16

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
            self.config = faiss.GpuIndexFlatConfig()
            self.config.device = self.device
            self.config.useFloat16 = self.use_fp16
            if use_ip:
                self.index = faiss.GpuIndexFlatIP(self.res, d, self.config)
            else:
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

    def _class_counts(self, class_idx: np.ndarray) -> np.ndarray:
        """Per-row class histogram via scatter-add."""
        n = class_idx.shape[0]
        counts = np.zeros((n, self.n_classes), dtype=np.int64)  # type: ignore[arg-type]
        np.add.at(counts, (np.arange(n)[:, None], class_idx), 1)
        return counts

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict int labels given X."""
        X = self._prepare(X)
        _, idx = self.index.search(X, k=self.n_neighbors)  # type: ignore[missing-argument]
        class_idx = self.y[idx]
        return np.argmax(self._class_counts(class_idx), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict float probabilities for labels given X."""
        X = self._prepare(X)
        _, idx = self.index.search(X, k=self.n_neighbors)  # type: ignore[missing-argument]
        class_idx = self.y[idx]
        return self._class_counts(class_idx) / self.n_neighbors


class FaissKNNMultilabelClassifier(FaissKNNClassifier):
    """A multilabel exact KNN classifier implemented using the FAISS library."""

    def _neighbor_label_sums(self, X: np.ndarray) -> np.ndarray:
        """Per-query, per-label count of positive (``==1``) neighbors."""
        X = self._prepare(X)
        _, idx = self.index.search(X, k=self.n_neighbors)  # type: ignore[missing-argument]
        # self.y[idx] -> (N, k, L) with values in {0, 1}; sum over k gives
        # count of 1s per (query, label).
        return self.y[idx].sum(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict one-hot int labels given X."""
        ones = self._neighbor_label_sums(X)
        # Matches argmax([zeros, ones]) tie-to-zero behavior of the old impl.
        return (ones > self.n_neighbors - ones).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict float probabilities for labels given X."""
        return self._neighbor_label_sums(X) / self.n_neighbors
