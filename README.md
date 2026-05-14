# faissknn

[![DOI](https://zenodo.org/badge/644113143.svg)](https://doi.org/10.5281/zenodo.18370747)

`faissknn` contains implementations for both multiclass and multilabel K-Nearest Neighbors Classifier implementations. The classifiers follow the `scikit-learn`: `fit`, `predict`, and `predict_proba` methods.

### Install

```bash
pip install faissknn
```

Pulls in [`faiss-cuda-cu128`](https://pypi.org/project/faiss-cuda-cu128/) (Taylor Geospatial's GPU-enabled FAISS wheels for CUDA 12.8) along with `numpy` and `torch`. No system CUDA toolkit required — the runtime libraries come from `nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12` on PyPI.

The default wheel works on:
- **CPU-only hosts** — `import faiss` succeeds, `faiss.get_num_gpus()` returns 0, all CPU index types work
- **CUDA 12.x hosts** (NVIDIA driver R525+) — full GPU acceleration
- **CUDA 13 hosts** (NVIDIA driver R580+) — via NVIDIA's forward-compat guarantee. You just don't get the `sm_100` (Blackwell) arch

#### Blackwell users (B100 / B200)

If you need `sm_100` baked in, use the CUDA 13 wheel:

```bash
pip install "faissknn[cu13]"
pip uninstall -y faiss-cuda-cu128
pip install --force-reinstall faiss-cuda
```

The `[cu13]` extra adds [`faiss-cuda`](https://pypi.org/project/faiss-cuda/) to the resolution, but because both packages ship the same `faiss/` module the manual uninstall+reinstall is what actually gives you a clean install. uv/pip can't auto-detect the host CUDA driver, so this is a one-time manual choice.

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
