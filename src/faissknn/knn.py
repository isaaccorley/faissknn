"""FAISS-based KNN classifiers for multiclass and multilabel classification."""

from typing import Any, Literal, Self

import numpy as np

try:
    import faiss
except ModuleNotFoundError as e:  # pragma: no cover
    # faissknn intentionally pins no FAISS backend (the cpu/cuda wheels share
    # the same `faiss` module and can't coexist). Turn the bare ImportError
    # into an actionable message naming the extras the user must choose from.
    msg = (
        "faissknn requires a FAISS backend, which is not installed. "
        "Install exactly one of the optional extras:\n"
        "  pip install 'faissknn[cpu]'   # CPU — all platforms (the default)\n"
        "  pip install 'faissknn[cuda]'  # GPU — CUDA 12.x, Linux x86_64\n"
        "  pip install 'faissknn[cu13]'  # GPU — CUDA 13 / Blackwell, Linux"
    )
    raise ModuleNotFoundError(msg) from e

# torch is a hard runtime dependency of this package, but be defensive in
# case anyone uses faissknn in a torch-free env (e.g. mocked imports).
try:
    import faiss.contrib.torch_utils  # monkey-patches faiss add/search
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

Metric = Literal["l2", "ip", "cosine"]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize; safe against zero rows."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


class FaissKNNClassifier:
    """A multiclass exact KNN classifier implemented using the FAISS library.

    Accepts both NumPy arrays and PyTorch tensors for ``fit`` and ``predict``.
    When a CUDA tensor is passed and ``device != "cpu"``, the FAISS search
    runs directly against GPU memory — no CPU round-trip.

    Parameters
    ----------
    n_neighbors : int
        Number of KNN neighbors.
    n_classes : int, optional
        Number of dataset classes (otherwise derived from the data).
    device : str, default="cpu"
        A torch device, e.g. cpu, cuda, cuda:0, etc.
    metric : {"l2", "ip", "cosine"}, default="l2"
        Distance metric used to rank neighbors:
          - "l2"     : squared Euclidean (FAISS IndexFlatL2)
          - "ip"     : inner product (FAISS IndexFlatIP)
          - "cosine" : cosine similarity; equivalent to "ip" but inputs are
                       L2-normalized on fit and predict so the user does not
                       have to.
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

    def _as_index_input(self, X: Any) -> Any:
        """Coerce input for FAISS ingestion + apply metric-specific prep.

        For CUDA torch tensors on a GPU index we return the tensor as-is
        (``faiss.contrib.torch_utils`` makes ``index.search`` accept it
        without copying through host memory); cosine metric normalization
        is applied in-place. Everything else becomes a contiguous fp32
        numpy array.
        """
        if _TORCH_AVAILABLE and isinstance(X, torch.Tensor):
            X = X.detach().contiguous().to(torch.float32)
            if self.cuda and X.is_cuda:
                if self.metric == "cosine":
                    X = torch.nn.functional.normalize(X, p=2, dim=1)
                return X
            X = X.cpu().numpy()
        X = np.atleast_2d(X).astype(np.float32)
        X = np.ascontiguousarray(X)
        if self.metric == "cosine":
            X = _l2_normalize(X)
        return X

    @staticmethod
    def _idx_to_numpy(idx: Any) -> np.ndarray:
        """FAISS may return a torch tensor when given torch input — normalize."""
        if _TORCH_AVAILABLE and isinstance(idx, torch.Tensor):
            return idx.cpu().numpy()
        return idx

    def create_index(self, d: int) -> None:
        """Create the faiss index.

        faiss exposes its GPU and CPU index classes through SWIG; ty can't
        always introspect them as guaranteed module members. We silence
        the per-attribute warnings on the construction calls below.

        Parameters
        ----------
        d : int
            Vector dimensionality of the index.
        """
        use_ip = self.metric in ("ip", "cosine")
        if self.cuda:
            self.res = faiss.StandardGpuResources()  # ty: ignore[possibly-missing-attribute]
            self.config = faiss.GpuIndexFlatConfig()  # ty: ignore[possibly-missing-attribute]
            # In the GPU branch self.device is always an int ordinal (set in
            # __init__); default to 0 to satisfy the typed config.device field.
            self.config.device = self.device if isinstance(self.device, int) else 0
            self.config.useFloat16 = self.use_fp16
            if use_ip:
                self.index = faiss.GpuIndexFlatIP(self.res, d, self.config)  # ty: ignore[possibly-missing-attribute]
            else:
                self.index = faiss.GpuIndexFlatL2(self.res, d, self.config)  # ty: ignore[possibly-missing-attribute]
        else:
            self.index = faiss.IndexFlatIP(d) if use_ip else faiss.IndexFlatL2(d)  # ty: ignore[possibly-missing-attribute]

    def fit(self, X: Any, y: Any) -> Self:
        """Store train X and y.

        ``y`` accepts ``np.ndarray`` or ``torch.Tensor`` — tensors are
        detached and converted to NumPy in this method.

        Parameters
        ----------
        X : ndarray or torch.Tensor
            Training feature matrix, shape (n_samples, n_features).
        y : ndarray or torch.Tensor
            Training labels, shape (n_samples,) for multiclass or
            (n_samples, n_labels) for multilabel.

        Returns
        -------
        Self
            The fitted classifier, for method chaining.
        """
        X = self._as_index_input(X)
        self.create_index(X.shape[-1])
        self.index.add(X)  # type: ignore[arg-type]
        if _TORCH_AVAILABLE and isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        self.y = y.astype(int)
        if self.n_classes is None:
            self.n_classes = len(np.unique(self.y))
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

    def predict(self, X: Any) -> np.ndarray:
        """Predict int labels given X.

        Parameters
        ----------
        X : ndarray or torch.Tensor
            Query feature matrix, shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Predicted integer class labels, shape (n_samples,).
        """
        X = self._as_index_input(X)
        _, idx = self.index.search(X, k=self.n_neighbors)  # type: ignore[missing-argument]
        idx = self._idx_to_numpy(idx)
        class_idx = self.y[idx]
        return np.argmax(self._class_counts(class_idx), axis=1)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict float probabilities for labels given X.

        Parameters
        ----------
        X : ndarray or torch.Tensor
            Query feature matrix, shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Predicted class probabilities, shape (n_samples, n_classes).
        """
        X = self._as_index_input(X)
        _, idx = self.index.search(X, k=self.n_neighbors)  # type: ignore[missing-argument]
        idx = self._idx_to_numpy(idx)
        class_idx = self.y[idx]
        return self._class_counts(class_idx) / self.n_neighbors


class FaissKNNMultilabelClassifier(FaissKNNClassifier):
    """A multilabel exact KNN classifier implemented using the FAISS library."""

    def _neighbor_label_sums(self, X: Any) -> np.ndarray:
        """Per-query, per-label count of positive (``==1``) neighbors."""
        X = self._as_index_input(X)
        _, idx = self.index.search(X, k=self.n_neighbors)  # type: ignore[missing-argument]
        idx = self._idx_to_numpy(idx)
        # self.y[idx] -> (N, k, L) with values in {0, 1}; sum over k gives
        # count of 1s per (query, label).
        return self.y[idx].sum(axis=1)

    def predict(self, X: Any) -> np.ndarray:
        """Predict one-hot int labels given X.

        Parameters
        ----------
        X : ndarray or torch.Tensor
            Query feature matrix, shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Predicted multilabel indicator matrix, shape
            (n_samples, n_labels), values in {0, 1}.
        """
        ones = self._neighbor_label_sums(X)
        # Matches argmax([zeros, ones]) tie-to-zero behavior of the old impl.
        return (ones > self.n_neighbors - ones).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict float probabilities for labels given X.

        Parameters
        ----------
        X : ndarray or torch.Tensor
            Query feature matrix, shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Per-label positive-vote fraction, shape (n_samples, n_labels),
            values in ``[0, 1]``.
        """
        return self._neighbor_label_sums(X) / self.n_neighbors
