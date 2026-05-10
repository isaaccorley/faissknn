"""FAISS-based KNN classifiers for multiclass and multilabel classification."""

from typing import Any

import faiss
import numpy as np

# torch is a hard runtime dependency of this package, but be defensive in
# case anyone uses faissknn in a torch-free env (e.g. mocked imports).
try:
    import torch

    import faiss.contrib.torch_utils  # noqa: F401  # monkey-patches faiss add/search

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


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

    Accepts both NumPy arrays and (CUDA-resident) PyTorch tensors for ``fit``
    and ``predict``. When a CUDA tensor is passed and ``device != "cpu"``, the
    FAISS search runs directly against the GPU memory — no CPU round-trip.
    """

    def __init__(
        self,
        n_neighbors: int,
        n_classes: int | None = None,
        device: str = "cpu",
    ) -> None:
        """Instantiate a faiss KNN Classifier."""
        self.n_neighbors = n_neighbors
        self.n_classes = n_classes

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
        """Coerce input to something the FAISS index can ingest efficiently.

        For CUDA torch tensors on a GPU index we return the tensor as-is
        (``faiss.contrib.torch_utils`` makes ``index.search`` accept it
        without copying through host memory). Everything else becomes a
        contiguous fp32 numpy array.
        """
        if _TORCH_AVAILABLE and isinstance(X, torch.Tensor):
            X = X.detach().contiguous().to(torch.float32)
            if self.cuda and X.is_cuda:
                return X
            X = X.cpu().numpy()
        X = np.atleast_2d(X).astype(np.float32)
        return np.ascontiguousarray(X)

    @staticmethod
    def _idx_to_numpy(idx: Any) -> np.ndarray:
        """FAISS may return a torch tensor when given torch input — normalize."""
        if _TORCH_AVAILABLE and isinstance(idx, torch.Tensor):
            return idx.cpu().numpy()
        return idx

    def create_index(self, d: int) -> None:
        """Create the faiss index."""
        if self.cuda:
            self.res = faiss.StandardGpuResources()  # type: ignore[possibly-missing-attribute]
            self.config = faiss.GpuIndexFlatConfig()
            self.config.device = self.device
            self.index = faiss.GpuIndexFlatL2(self.res, d, self.config)
        else:
            self.index = faiss.IndexFlatL2(d)

    def fit(self, X: Any, y: np.ndarray) -> object:
        """Store train X and y."""
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

    def predict(self, X: Any) -> np.ndarray:
        """Predict int labels given X."""
        X = self._as_index_input(X)
        _, idx = self.index.search(X, k=self.n_neighbors)  # type: ignore[missing-argument]
        idx = self._idx_to_numpy(idx)
        class_idx = self.y[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes),  # type: ignore[invalid-argument-type]
            axis=1,
            arr=class_idx.astype(np.int16),
        )
        return np.argmax(counts, axis=1)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict float probabilities for labels given X."""
        X = self._as_index_input(X)
        _, idx = self.index.search(X, k=self.n_neighbors)  # type: ignore[missing-argument]
        idx = self._idx_to_numpy(idx)
        class_idx = self.y[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes),  # type: ignore[invalid-argument-type]
            axis=1,
            arr=class_idx.astype(np.int16),
        )
        return counts / self.n_neighbors


class FaissKNNMultilabelClassifier(FaissKNNClassifier):
    """A multilabel exact KNN classifier implemented using the FAISS library."""

    def predict(self, X: Any) -> np.ndarray:
        """Predict one-hot int labels given X."""
        X = self._as_index_input(X)
        _, idx = self.index.search(X, k=self.n_neighbors)  # type: ignore[missing-argument]
        idx = self._idx_to_numpy(idx)
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

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict float probabilities for labels given X."""
        X = self._as_index_input(X)
        _, idx = self.index.search(X, k=self.n_neighbors)  # type: ignore[missing-argument]
        idx = self._idx_to_numpy(idx)
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
