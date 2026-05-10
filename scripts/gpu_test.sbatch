#!/bin/bash
# Run the faissknn test suite on a TGI RAILS A100 with --device=cuda.
#
# Usage:
#   sbatch --account=<PROJECT> scripts/gpu_test.sbatch <git-ref>
#
# `<git-ref>` is anything `git checkout` accepts (branch name, tag, commit).
# The job clones the repo into the per-job /tmp scratch, installs the
# project (plus dev deps), and runs `pytest --device=cuda`.
#
# faiss-cuda-cu128 is manylinux_2_34 — RAILS (RHEL 8 / glibc 2.28) refuses
# to install it via pip's normal path, so we unzip the wheel directly into
# site-packages and rely on spack gcc 11.4's libstdc++ on LD_LIBRARY_PATH.
#
#SBATCH --job-name=faissknn-gputest
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
mkdir -p logs

GIT_REF="${1:?usage: gpu_test.sbatch <git-ref>}"
REPO_URL="${REPO_URL:-https://github.com/isaaccorley/faissknn.git}"

WORK="$(mktemp -d "/tmp/faissknn-gputest-${SLURM_JOB_ID:-$$}-XXX")"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

echo "==> Cloning $REPO_URL ($GIT_REF) on $(hostname)"
git clone --depth 50 "$REPO_URL" repo
cd repo
git checkout "$GIT_REF"
git log -1 --oneline

source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module unload cuda 2>/dev/null || true

# spack gcc 11.4 provides the libstdc++ symbols (GLIBCXX_3.4.29) that the
# manylinux_2_34 wheel needs at runtime; RAILS' system libstdc++ is too old.
GCC_LIBDIR="/sw/spack/v1/apps/gcc/11.4.0-gcc-8.5.0-d6zlqss/lib64"
[ -d "$GCC_LIBDIR" ] && export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}"

PY_VERSION="${PY_VERSION:-3.12}"
uv venv --python "$PY_VERSION" .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Resolve faiss-cuda-cu128 wheel URL (matching PY tag), grab via curl, unzip
# directly into site-packages. Bypasses pip's manylinux platform check.
PY_TAG="cp${PY_VERSION//./}"
echo "==> Locating faiss-cuda-cu128 wheel for $PY_TAG"
WHL_URL=$(python - <<PY
import json, urllib.request
data = json.loads(urllib.request.urlopen("https://pypi.org/pypi/faiss-cuda-cu128/json").read())
url = next(u["url"] for u in data["urls"] if "$PY_TAG" in u["filename"])
print(url)
PY
)
echo "    $WHL_URL"
curl -sL "$WHL_URL" -o faiss-wheel.whl

SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "==> Installing runtime deps via pip"
python -m ensurepip --upgrade >/dev/null
python -m pip install --quiet numpy "torch>=2,<3" packaging \
    "nvidia-cuda-runtime-cu12>=12.8" "nvidia-cublas-cu12>=12.8"
echo "==> Unzipping wheel into $SITE"
unzip -oq faiss-wheel.whl -d "$SITE"

echo "==> Installing faissknn + dev deps from $GIT_REF"
python -m pip install --quiet ".[dev]" || python -m pip install --quiet . \
    pytest pytest-cov scikit-learn

echo "==> Sanity import"
python - <<'PY'
import faiss, torch
print("faiss", faiss.__version__, "| num gpus:", faiss.get_num_gpus())
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
PY

echo "==> Running pytest --device=cuda"
pytest -vv --device=cuda tests/
echo "==> GPU tests passed for $GIT_REF"
