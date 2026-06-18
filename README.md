# faissknn

[![DOI](https://zenodo.org/badge/644113143.svg)](https://doi.org/10.5281/zenodo.18370747)

`faissknn` contains implementations for both multiclass and multilabel K-Nearest Neighbors Classifier implementations. The classifiers follow the `scikit-learn`: `fit`, `predict`, and `predict_proba` methods.

### Install

`faissknn` needs a FAISS backend, which you choose at install time via an extra. Pick **exactly one** — they ship the same `faiss` module and can't coexist. For CPU (works everywhere — macOS, Windows, Linux; no GPU, CUDA driver, or system toolkit required):

```bash
pip install "faissknn[cpu]"
```

This pulls in [`faiss-cpu`](https://pypi.org/project/faiss-cpu/) along with `numpy` and `torch`.

> A bare `pip install faissknn` installs no FAISS backend and raises a clear error at import time telling you to pick an extra. Always install one of `[cpu]`, `[cuda]`, or `[cu13]`.

#### GPU acceleration (CUDA 12.x)

For GPU-enabled FAISS on CUDA 12.x hosts (NVIDIA driver R525+), use the `[cuda]` extra, which installs [`faiss-cuda-cu128`](https://pypi.org/project/faiss-cuda-cu128/) (Taylor Geospatial's GPU wheels for CUDA 12.8) instead of `faiss-cpu`. No system CUDA toolkit needed — the runtime libraries come from `nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12` on PyPI.

```bash
pip install "faissknn[cuda]"
```

These wheels (Linux x86_64) also run on CUDA 13 hosts (driver R580+) via NVIDIA's forward-compat guarantee — you just don't get the `sm_100` (Blackwell) arch.

#### Blackwell users (B100 / B200)

If you need `sm_100` baked in, use the CUDA 13 wheel via the `[cu13]` extra, which installs [`faiss-cuda`](https://pypi.org/project/faiss-cuda/):

```bash
pip install "faissknn[cu13]"
```

uv/pip can't auto-detect the host CUDA driver, so the backend is a manual choice. Because nothing is installed until you pick an extra, a **fresh** install of any single extra is clean — no base `faiss-cpu` to fight, no uninstall/reinstall dance. If you later want to **switch** backends in the same environment, uninstall the current one first (e.g. `pip uninstall -y faiss-cpu`) before installing the other extra, since the FAISS packages share the `faiss` module and can't coexist.

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
