# faissknn

[![DOI](https://zenodo.org/badge/644113143.svg)](https://doi.org/10.5281/zenodo.18370747)

`faissknn` contains implementations for both multiclass and multilabel K-Nearest Neighbors Classifier implementations. The classifiers follow the `scikit-learn`: `fit`, `predict`, and `predict_proba` methods.

### Install

The FAISS authors recommend to install `faiss` through conda e.g. `conda install -c pytorch faiss-gpu`. See [FAISS install page](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for more info.

Once `faiss` is installed, `faissknn` can be installed through pypi:

```
pip install faissknn
```

### Usage

Multiclass:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from faissknn import FaissKNNClassifier

x, y = make_classification()
x_train, x_test, y_train, y_test = train_test_split(x, y)
model = FaissKNNClassifier(
    n_neighbors=5,
    n_classes=None,
    device="cpu"
)
model.fit(x_train, y_train)

y_pred = model.predict(x_test) # (N,)
y_proba = model.predict_proba(x_test) # (N, C)
```

Multilabel:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from faissknn import FaissKNNMultilabelClassifier

x, y = make_multilabel_classification()
x_train, x_test, y_train, y_test = train_test_split(x, y)
model = FaissKNNMultilabelClassifier(
    n_neighbors=5,
    device="cpu"
)
model.fit(x_train, y_train)

y_pred = model.predict(x_test) # (N, C)
y_proba = model.predict_proba(x_test) # (N, C)
```

GPU/CUDA: `faissknn` also supports running on the GPU to speed up computation. Simply change the device to `cuda` or a specific cuda device `cuda:0`

```python
model = FaissKNNClassifier(
    n_neighbors=5,
    device="cuda"
)
model = FaissKNNClassifier(
    n_neighbors=5,
    device="cuda:0"
)
```

### Cite

If you use `faissknn` in your research, please considering citing!

```bibtex
@software{isaac_corley_2026_18370748,
  author       = {Isaac Corley},
  title        = {isaaccorley/faissknn: Zenodo Cite},
  month        = jan,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.0.3},
  doi          = {10.5281/zenodo.18370748},
  url          = {https://doi.org/10.5281/zenodo.18370748},
}
```
