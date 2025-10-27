"""FAISS-based KNN classifiers for multiclass and multilabel classification."""

import faiss
import numpy as np


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
    """

    def __init__(self, n_neighbors: int, n_classes: int | None = None, device: str = "cpu") -> None:
        """Instantiate a faiss KNN Classifier.

        Parameters
        ----------
        n_neighbors : int
            Number of KNN neighbors
        n_classes : int, optional
            Number of dataset classes (otherwise derived from the data)
        device : str, default="cpu"
            A torch device, e.g. cpu, cuda, cuda:0, etc.
        """
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

    def create_index(self, d: int) -> None:
        """Create the faiss index.

        Parameters
        ----------
        d : int
            Feature dimension
        """
        if self.cuda:
            self.res = faiss.StandardGpuResources()  # type: ignore[possibly-missing-attribute]
            self.config = faiss.GpuIndexFlatConfig()
            self.config.device = self.device
            self.index = faiss.GpuIndexFlatL2(self.res, d, self.config)
        else:
            self.index = faiss.IndexFlatL2(d)

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        """Store train X and y.

        Parameters
        ----------
        X : np.ndarray
            Input features (N, d)
        y : np.ndarray
            Input labels (N, ...)

        Returns
        -------
        FaissKNNClassifier
            The fitted classifier instance
        """
        X = np.atleast_2d(X).astype(np.float32)
        X = np.ascontiguousarray(X)
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
        """Predict int labels given X.

        Parameters
        ----------
        X : np.ndarray
            Input features (N, d)

        Returns
        -------
        np.ndarray
            Predicted labels (N,)
        """
        X = np.atleast_2d(X).astype(np.float32)
        _, idx = self.index.search(X, k=self.n_neighbors)  # type: ignore[missing-argument]
        class_idx = self.y[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes),  # type: ignore[invalid-argument-type]
            axis=1,
            arr=class_idx.astype(np.int16),
        )
        return np.argmax(counts, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict float probabilities for labels given X.

        Parameters
        ----------
        X : np.ndarray
            Input features (N, d)

        Returns
        -------
        np.ndarray
            Predicted probabilities per label (N, c)
        """
        X = np.atleast_2d(X).astype(np.float32)
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
        """Predict one-hot int labels given X.

        Parameters
        ----------
        X : np.ndarray
            Input features (N, d)

        Returns
        -------
        np.ndarray
            Predicted labels (N, c)
        """
        X = np.atleast_2d(X).astype(np.float32)
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
        """Predict float probabilities for labels given X.

        Parameters
        ----------
        X : np.ndarray
            Input features (N, d)

        Returns
        -------
        np.ndarray
            Predicted probabilities per label (N, c)
        """
        X = np.atleast_2d(X).astype(np.float32)
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
