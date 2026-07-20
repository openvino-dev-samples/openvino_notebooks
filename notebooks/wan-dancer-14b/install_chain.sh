#!/usr/bin/env bash
# Bootstrap a clean venv for the wan-dancer-14b notebook.
#
# This file documents every pip command we tested end-to-end on a fresh
# Python 3.12 venv on this box (Flex 170 + custom-built OpenVINO). Use the
# `commands` array as the single source of truth when you paste install
# cells into the notebook.
set -euo pipefail

VENV="${VENV:-/home/ethan/intel/venv-wan-dancer}"
OV_WHEEL="/home/ethan/intel/openvino/build/wheels/openvino-2026.3.0-21416-cp312-cp312-manylinux_2_39_x86_64.whl"

if [[ ! -d "$VENV" ]]; then
    python3.12 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install -U pip wheel setuptools "setuptools<81"

# 1) Custom OpenVINO wheel (the one built from source for this box).
pip install --no-deps "$OV_WHEEL"

# 2) optimum-intel editable (matches the wan2.2 notebook).
pip install --no-deps -e /home/ethan/intel/optimum-intel

# 3) Common runtime deps with --no-deps to keep our OV/transformers pinned.
#    transformers must stay at 4.46.2 because DiffSynth uses removed APIs
#    in 5.x; huggingface_hub 0.30.x is the last with `DDUFEntry`.
NNCF_DEPS=(
  sympy mpmath numpy huggingface_hub tokenizers certifi idna sniffio
  annotated-types pydantic typing_extensions urllib3 charset-normalizer
  requests pillow regex pyyaml filelock tqdm fsspec aiohttp
  lazy_loader joblib numba scipy soundfile pooch decorator networkx
  msgpack soxr audioread proglog python-dotenv imageio-ffmpeg
  optimum wcwidth av opencv-python-headless loguru
  rich tabulate scikit-learn matplotlib dill referencing rpds-py
  pynvml beautifulsoup4 markdown pyparsing six python-dateutil pytz
  tzdata jsonpath-ng dpath duckdb narwhals pandas attrs
  jsonschema_specifications tomlkit ruff ffmpy aiofiles altair
  toolz pydub semantic_version httpx importlib_resources
  fastjsonschema traitlets gradio_client jupyter_client jupyter_core
  nbformat nbconvert ipykernel
)
# NB: `--no-deps` is required because we already install what we need at
# the right version; pip complains a lot but doesn't actually break.
pip install --no-deps "${NNCF_DEPS[@]}" 2>&1 | grep -v "ERROR: " || true

# Lock specific versions needed to keep transformers 4.46.2 happy
# (transformers pinned to 4.46.2 because DiffSynth uses removed APIs in 5.x;
# huggingface_hub 0.30.2 is the last 0.x with DDUFEntry).
pip install --no-deps \
  "transformers==4.46.2" \
  "tokenizers==0.20.3" \
  "huggingface_hub==0.30.2"

# 4) Model libs, also --no-deps to keep things frozen.
pip install --no-deps \
  "torch==2.7.0" "torchvision==0.22.0" "torchaudio==2.7.0" \
  --index-url https://download.pytorch.org/whl/cpu

pip install --no-deps "diffusers==0.34.0" \
  safetensors accelerate sentencepiece peft ftfy \
  "gradio==4.19.2" moviepy librosa ffmpeg-python \
  nncf modelscope_hub modelscope einops easydict addict

# 5) Install the upstream Wan-Dancer source tree (it is NOT published on
#    PyPI; we install via `setup.py install` because pip's pep517 isolation
#    breaks the upstream ``pkg_resources`` import).
git clone --depth 1 https://github.com/Wan-Video/Wan-Dancer.git /tmp/wan-dancer-upstream
python /tmp/wan-dancer-upstream/setup.py install

echo "✅ Wan-Dancer OV venv ready at $VENV"
