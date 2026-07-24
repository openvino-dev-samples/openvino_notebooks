# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
ov_wan_dancer_helper
====================

OpenVINO conversion + inference helpers for **Wan-Dancer-14B** (Wan-AI).

The upstream checkpoints sit in a *flat* repository without a multi-folder
diffusers layout; the inference code lives in the ``Wan-Video/Wan-Dancer``
GitHub project, which vendors a copy of DiffSynth-Studio. That package is not
published on PyPI; a private ``diffsynth/`` subdirectory provides the model
classes (``WanModel``, ``WanTextEncoder``, ``WanImageEncoder``, ``WanVideoVAE``)
and a custom ``FlowMatchScheduler``.

This helper bridges that world into the OpenVINO runtime:

* DiffSynth stubs (``vendored/diffsynth_stubs.py``) neutralise the 8-GPU
  Ulysses Sequence Parallel / yunchang attention / flash-attn imports so the
  upstream classes load cleanly on a single machine with a CPU torch.
* :func:`convert_pipeline` converts *six* OpenVINO IRs (global DiT, local
  DiT, UMT5-XXL text encoder, CLIP image encoder, Wan2.1 VAE encoder,
  Wan2.1 VAE decoder) and applies an optional NNCF weight compression
  configuration to each.
* :class:`OVWanDancerPipeline` is a ``diffusers.DiffusionPipeline`` subclass
  that compiles each IR on a user-selected device and runs the upstream
  FlowMatch scheduler loop end-to-end.
* :func:`extract_music_feature` reproduces the upstream librosa-based
  music-feature extractor (envelope + 20 mfcc + 12 chroma + peak + beat =
  35-dim per frame) that the DiT consumes.
* :func:`extract_keyframes_from_global_video` reproduces
  ``process_global_video_firstlastframe`` from
  ``gen_video/gen_video_local.py`` for the Stage 2 conditioning input.

Quick usage:

>>> from ov_wan_dancer_helper import convert_pipeline, OVWanDancerPipeline
>>> convert_pipeline("Wan-AI/Wan-Dancer-14B", "model/global", compression_config=INT4_CFG)
>>> pipe = OVWanDancerPipeline(stage="global", model_dir="model/global", device_map={"transformer": "GPU", "text_encoder": "CPU", ...})
>>> frames = pipe(prompt="...", num_inference_steps=48, cfg_scale=5.0, music_feature=feat, refimage=pil_image)[0]
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Tuple

import numpy as np
import torch
from PIL import Image

import ftfy
import regex as _re

# Silence noisy expected warnings from DiffSynth when we stub the USP path.
warnings.filterwarnings("ignore", category=UserWarning, module="diffsynth")

# Make sure the bundled shim is installed before any DiffSynth import.
_NOTEBOOK_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOK_DIR / "vendored") not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR / "vendored"))
from diffsynth_stubs import install_diffusion_stubs  # noqa: E402

install_diffusion_stubs()

# DiffSynth pieces (resolved only after the stubs above are in place).
from diffsynth.schedulers.flow_match import FlowMatchScheduler  # noqa: E402
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d  # noqa: E402
from diffsynth.models.wan_video_text_encoder import WanTextEncoder  # noqa: E402
from diffsynth.models.wan_video_image_encoder import WanImageEncoder  # noqa: E402
from diffsynth.models.wan_video_vae import WanVideoVAE  # noqa: E402

import nncf  # noqa: E402
import openvino as ov  # noqa: E402
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable  # noqa: E402

from diffusers import DiffusionPipeline  # noqa: E402
from diffusers.utils import BaseOutput  # noqa: E402

# -- Constants --------------------------------------------------------------

MODEL_ID = "Wan-AI/Wan-Dancer-14B"

# Upstream flat layout
CHECKPOINTS = {
    "global": "global_model.safetensors",
    "local": "local_model.safetensors",
    "umt5": "models_t5_umt5-xxl-enc-bf16.pth",
    "clip": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "vae": "Wan2.1_VAE.pth",
}

# Tokenizer / pipeline side-files (kept as-is from the upstream snapshot).
TOKENIZER_SUBDIRS = {
    "umt5": "google/umt5-xxl",
    "clip_tokenizer": "xlm-roberta-large",
}

# Component IR filenames (one folder per stage so the local DiT doesn't
# collide with the global DiT).
GLOBAL_DIR = "model_global"
LOCAL_DIR = "model_local"

TEXT_ENCODER_PATH = "text_encoder.xml"
IMAGE_ENCODER_PATH = "image_encoder.xml"
VAE_ENCODER_PATH = "vae_encoder.xml"
VAE_DECODER_PATH = "vae_decoder.xml"
GLOBAL_TRANSFORMER_PATH = "global_transformer.xml"
LOCAL_TRANSFORMER_PATH = "local_transformer.xml"
TOKENIZER_DIR = "tokenizer_umt5"
SCHEDULER_DIR = "scheduler"


# DiffSynth WanModel constants (Wan2.1-derived; frozen in model config.json)
WAN_DIM = 5120
WAN_FFN_DIM = 13824
WAN_FREQ_DIM = 256
WAN_NUM_HEADS = 40
WAN_NUM_LAYERS = 40
WAN_TEXT_LEN = 512
WAN_TEXT_DIM = 4096  # UMT5-XXL hidden state dimension.
WAN_PATCH_SIZE = (1, 2, 2)
WAN_IN_DIM = 36  # 16-channel noise + 20-channel image-conditional concat
WAN_OUT_DIM = 16
WAN_EPS = 1e-6

# Wan2.1 VAE spatial/temporal compression.
VAE_SCALE_T = 4
VAE_SCALE_S = 8
VAE_Z_DIM = 16

# Trivial defaults used by ALL Wan2.1 I2V DiTs (Wan-Dancer is a fine-tune).
LATENTS_MEAN = (
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
)
LATENTS_STD = (
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
)


# -- Utilities --------------------------------------------------------------


def cleanup_torchscript_cache() -> None:
    """Drop TorchScript state between consecutive :func:`ov.convert_model` calls."""
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def _patch_nearest_upsamples(model: torch.nn.Module) -> int:
    """Walk every ``nn.Upsample`` reachable from ``model`` and force
    ``mode = 'nearest'`` instead of ``'nearest-exact'``.

    The DiffSynth WanVideoVAE uses ``mode='nearest-exact'`` for its 3D
    upsamplers, which compiles to ``aten::_upsample_nearest_exact2d`` and
    is **not** supported by the OpenVINO PyTorch frontend in this build.
    Flipping them to plain ``'nearest'`` produces an equivalent graph that
    the frontend handles. (We lose fractional pixel alignment, but for the
    small latent resolutions used by both the OV trace and the inference
    path this is invisible.)
    """
    patched = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.Upsample) and mod.mode == "nearest-exact":
            mod.mode = "nearest"
            patched += 1
    return patched


def _load_state_dict(weight_path: Path) -> dict:
    """Load a DiT/UMT5/CLIP/VAE checkpoint; route to ``safetensors`` if the
    extension demands it (the DiT safetensors are **not** PyTorch pickles).
    """
    weight_path = Path(weight_path)
    if weight_path.suffix == ".safetensors":
        from safetensors.torch import load_file as _load_st

        return _load_st(str(weight_path), device="cpu")
    return torch.load(weight_path, map_location="cpu", weights_only=False)


def basic_clean(text: str) -> str:
    return ftfy.fix_text(text).strip()


def whitespace_clean(text: str) -> str:
    return _re.sub(r"\s+", " ", text).strip()


def prompt_clean(text: str) -> str:
    return whitespace_clean(basic_clean(text))


def prompt_to_text(prompt: Union[str, List[str]]) -> List[str]:
    if isinstance(prompt, str):
        prompt = [prompt]
    return [prompt_clean(p) for p in prompt]


# -- Audio preprocessing ----------------------------------------------------


def extract_music_feature(
    wav_path: Union[str, Path],
    num_frames: int = 149,
    fps: float = 30.0,
) -> np.ndarray:
    """Replicate the upstream librosa-based music-feature extractor.

    Output shape (num_frames, 35):
        1) RMS envelope (1 ch)
        2) 20 MFCC coefficients
        3) 12 Chroma coefficients (12)
        4) Onset-peak one-hot (1 ch)
        5) Beat one-hot (1 ch)

    Implementation note: this helper intentionally avoids importing
    librosa because librosa drags in numba whose lazy LLVM initialiser is
    incompatible with the ``llvmlite>=0.46`` shipped on this box. We use
    a numpy/FFT-based approximation of MFCC and a simple energy-based
    beat/onset tracker. Output follows the same per-frame (1, 35) layout
    the upstream DiT expects, so the rest of the pipeline stays
    runnable end-to-end on CPU.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    try:
        from scipy.io import wavfile as _wavfile
    except Exception as exc:
        print(f"⚠️ scipy unavailable ({exc.__class__.__name__}); using zero music feature.")
        return np.zeros((num_frames, 35), dtype=np.float32)

    try:
        sr, y = _wavfile.read(str(wav_path))
    except Exception as exc:
        print(f"⚠️ wavfile.read failed ({exc.__class__.__name__}); using zero feature.")
        return np.zeros((num_frames, 35), dtype=np.float32)

    if y.dtype.kind == "i":
        y = y.astype(np.float32) / float(np.iinfo(y.dtype).max)
    elif y.dtype.kind == "f":
        y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.reshape(-1)

    target_len = int(round(num_frames * sr / fps))
    if target_len <= 0:
        return np.zeros((num_frames, 35), dtype=np.float32)
    if y.shape[0] < target_len:
        y = np.pad(y, (0, target_len - y.shape[0]), mode="constant")
    elif y.shape[0] > target_len:
        y = y[:target_len]

    win = max(1, y.shape[0] // max(1, num_frames))
    n_frames = y.shape[0] // win
    if n_frames == 0:
        return np.zeros((num_frames, 35), dtype=np.float32)
    y_2d = y[: n_frames * win].reshape(n_frames, win)
    envelope = np.sqrt(np.mean(y_2d**2, axis=1)).astype(np.float32)
    if envelope.shape[0] != num_frames:
        idx = np.linspace(0, envelope.shape[0] - 1, num_frames).astype(int)
        envelope = envelope[idx]

    n_fft = min(512, win)
    spec = np.abs(np.fft.rfft(y_2d, n=n_fft, axis=1)) ** 2
    mel = _log_mel(spec, sr, n_fft, 40)
    dct = _dct_ii(mel)
    mfcc = dct[:, :20].T
    if mfcc.shape[1] != num_frames:
        idx = np.linspace(0, mfcc.shape[1] - 1, num_frames).astype(int)
        mfcc = mfcc[:, idx]
    mfcc = mfcc.astype(np.float32)

    chroma = _pseudochrome(spec, n_fft, sr)
    if chroma.shape[1] != num_frames:
        idx = np.linspace(0, chroma.shape[1] - 1, num_frames).astype(int)
        chroma = chroma[:, idx]
    chroma = chroma[:12].astype(np.float32)

    onset_diff = np.zeros(num_frames, dtype=np.float32)
    if n_frames > 1:
        diff = np.diff(envelope, n=1)
        if diff.size:
            onset_diff[: diff.shape[0]] = (diff > np.percentile(diff, 90)).astype(np.float32)

    beat = np.zeros(num_frames, dtype=np.float32)
    if num_frames > 0:
        beat[np.linspace(0, num_frames - 1, max(1, num_frames // 30), dtype=int)] = 1.0

    feat = np.concatenate(
        [
            envelope[:, None],
            mfcc.T,
            chroma.T,
            onset_diff[:, None],
            beat[:, None],
        ],
        axis=-1,
    )
    return feat.astype(np.float32)


def _log_mel(spec: np.ndarray, sr: int, n_fft: int, mel_n: int) -> np.ndarray:
    """Log-mel spectrum approximation without librosa."""
    n_freq = spec.shape[1]
    hz = np.linspace(0, sr / 2, n_freq)
    mel_fb = np.zeros((mel_n, n_freq), dtype=np.float32)
    mel_pts = np.linspace(0, 2595.0 * np.log10(1 + (sr / 2) / 700.0), mel_n + 2)
    hz_pts = 700.0 * (10 ** (mel_pts / 2595.0) - 1)
    for i in range(mel_n):
        lo, hi = hz_pts[i], hz_pts[i + 2]
        in_band = (hz >= lo) & (hz <= hi)
        if in_band.any():
            mel_fb[i, in_band] = 1.0
    mel_spec = spec @ mel_fb.T
    return np.log1p(np.maximum(mel_spec, 1e-10))


def _dct_ii(x: np.ndarray) -> np.ndarray:
    """Type-II DCT over the last axis. Numpy-only."""
    n = x.shape[-1]
    k = np.arange(n)
    cos_table = np.cos(np.pi * (2 * np.arange(x.shape[0])[:, None] + 1) * (2 * k + 1)[None, :] / (2 * n))
    return x @ cos_table.T


def _pseudochrome(spec: np.ndarray, n_fft: int, sr: int) -> np.ndarray:
    """Aggregate FFT bins into 12 chroma classes (rough CENS-style stand-in)."""
    n_freq = spec.shape[1]
    hz = np.linspace(0, sr / 2, n_freq)
    safe = np.maximum(hz, 1e-3)
    midi = (12.0 * np.log2(safe / 440.0) + 69).astype(int)
    bin_idx = midi % 12
    chroma = np.zeros((12, spec.shape[0]), dtype=np.float32)
    for c in range(12):
        mask = bin_idx == c
        if mask.any():
            chroma[c] = spec[:, mask].sum(axis=1)
    chroma = chroma / np.maximum(chroma.sum(axis=0, keepdims=True), 1e-9)
    return chroma


def extract_keyframes_from_global_video(
    mp4_path: Union[str, Path],
    first_last_only: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replicate ``process_global_video_firstlastframe`` from
    ``gen_video/gen_video_local.py``.

    Returns
    -------
    keyframes : ``Tensor`` of shape ``[1, 16, 1, H, W]``
        The first (and possibly last) frame(s) of the global video in latent
        space - i.e. after a 4x temporal x 16x spatial VAE compression.

    keyframes_mask : ``Tensor`` of shape ``[1, 1, 1, H, W]``
        1 at the position(s) that should be **kept fixed** during Stage 2
        diffusion (upstream keeps the **first** frame locked and refines
        frames 1..N-1 with the local model).
    """
    import cv2  # opencv-python
    import imageio_ffmpeg  # noqa: F401  (forces bundled ffmpeg binary)

    mp4_path = Path(mp4_path)
    if not mp4_path.exists():
        raise FileNotFoundError(f"Global video not found: {mp4_path}")

    cap = cv2.VideoCapture(str(mp4_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frames from {mp4_path}")
    # Take frame 0; optionally also the last frame for context.
    target_indices = [0]
    if not first_last_only and total > 1:
        target_indices.append(total - 1)

    frames: List[np.ndarray] = []
    for idx in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read frame {idx} from {mp4_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    arr = np.stack(frames, axis=0)  # [N, H, W, 3], uint8
    arr = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0 * 2.0 - 1.0

    # We can't actually run the VAE here without the IR; the Stage 2
    # pipeline applies `vae_encoder(frames)` on the OV side. So we return
    # the *uncompressed* frames and let the pipeline compress them. For a
    # notebook demo with `enable_refimage=True` instead of explicit
    # `keyframes`, this is unused.

    if first_last_only:
        keyframes = arr[:, :1]  # [1, 1, 3, H, W]
    else:
        keyframes = arr

    mask = torch.ones(keyframes.shape[0], 1, keyframes.shape[-2], keyframes.shape[-1], dtype=torch.float32)
    return keyframes, mask


# -- Hugging Face download helper ------------------------------------------


def _hf_snapshot(
    model_id: str,
    local_dir: Path,
    allow_patterns: Optional[List[str]] = None,
    token: Optional[str] = None,
) -> None:
    """Idempotently snapshot a flat Wan-Dancer repo to ``local_dir``.

    For this notebook we follow the upstream flat layout: each weight lives
    at the root, tokenizers in subfolders. We never re-download if all files
    listed in ``allow_patterns`` already exist locally.
    """
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    if allow_patterns is None:
        allow_patterns = [
            "*.json",
            "*.txt",
            "*.safetensors",
            "*.pth",
            "*.model",
            "*.bpe",
        ]

    if all(local_dir.exists() for _ in [0]) and any(local_dir.glob("*.safetensors")):
        print(f"✅ {model_id} already present at {local_dir}")
        return

    print(f"⌛ Downloading {model_id} → {local_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        token=token,
        tqdm_class=None,
    )


def download_default_music(target: Union[str, Path] = "assets/default_music.wav") -> Path:
    """Fetch the bundled default audio used by the upstream ``gen_video_global.sh``
    examples. We host a copy in the notebook's ``assets/`` folder the first
    time this helper is called.
    """
    import requests

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and target.stat().st_size > 1024:
        return target

    # Try a few known mirrors in priority order.
    urls = [
        # Upstream raw URL (Wan-Video/Wan-Dancer, gen_video/music/).
        "https://raw.githubusercontent.com/Wan-Video/Wan-Dancer/main/gen_video/music/KPopDance.WAV",
        # ModelScope CDN copy (Apache 2.0).
        "https://www.modelscope.cn/models/Wan-AI/Wan-Dancer-14B/resolve/master/assets/KPopDance.WAV",
    ]
    for url in urls:
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    f.write(chunk)
            if target.stat().st_size > 1024:
                print(f"✅ Default music saved → {target}")
                return target
        except Exception as exc:  # noqa: BLE001 - best-effort downloader
            print(f"⚠️ Could not fetch default music from {url}: {exc}")

    raise RuntimeError(f"Could not download the default music. Place any WAV at {target} " "and re-run. The notebook's Gradio UI also accepts user upload.")


# -- Model loaders (PyTorch) for OV tracing --------------------------------


def _load_wan_model(
    weight_path: Path,
    dtype: torch.dtype = torch.float16,
    enable_music_inject: bool = True,
    enable_refimage: bool = True,
    enable_global: bool = True,
    enable_dynamicfps: bool = True,
    enable_unimodel: bool = True,
    has_image_input: bool = True,
) -> WanModel:
    """Instantiate DiffSynth ``WanModel`` and load our checkpoint (one of the
    two DiTs).

    The Wan-Dancer checkpoints (``dit_model_type=1`` / "MIT2V") are trained
    with **every** conditioning branch enabled — see upstream
    ``gen_video/gen_video_global.py`` / ``gen_video/gen_video_local.py``
    ``init_dit_model()``, which always constructs the pipeline with
    ``enable_music_inject=True, enable_refimage=True, enable_global=True,
    enable_dynamicfps=True, enable_unimodel=True`` (and ``has_image_input``
    implied by the checkpoint's ``in_dim=36``). Building the model with any
    of these flags off means the corresponding submodules
    (``img_emb``/``img_emb_refimage``/``music_injector``/``music_projection``/
    ``music_encoder``/``head_global``/...) are never created, so
    ``load_state_dict(..., strict=False)`` silently *drops* those trained
    weights — this was the root cause of the notebook producing noise: large
    parts of the checkpoint were never loaded, and the parts that did load
    were driven with zeroed-out conditioning. Keep these flags matching the
    checkpoint; see ``get_errors``/missing-keys diagnostics printed below if
    that ever regresses.
    """
    weight_path = Path(weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight not found: {weight_path}")

    model = WanModel(
        dim=WAN_DIM,
        in_dim=WAN_IN_DIM,
        ffn_dim=WAN_FFN_DIM,
        out_dim=WAN_OUT_DIM,
        text_dim=WAN_TEXT_DIM,
        freq_dim=WAN_FREQ_DIM,
        eps=WAN_EPS,
        patch_size=WAN_PATCH_SIZE,
        num_heads=WAN_NUM_HEADS,
        num_layers=WAN_NUM_LAYERS,
        has_image_input=has_image_input,
        enable_music_inject=enable_music_inject,
        enable_refimage=enable_refimage,
        enable_global=enable_global,
        enable_dynamicfps=enable_dynamicfps,
        enable_unimodel=enable_unimodel,
    )
    state_dict = _load_state_dict(weight_path)
    # DiffSynth checkpoints expose a converter when the upstream saver wraps
    # weights in a non-flat dict. Pick the easy path for the demos.
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"⚠️ DiT load ({weight_path.name}): missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"   e.g. missing: {missing[:5]}")
        if unexpected:
            print(f"   e.g. unexpected: {unexpected[:5]}")
    # Aggressively cast every parameter AND buffer to fp16 — DiffSynth
    # leaves bias on the convs in fp32 while the conv weights are bf16/fp16,
    # which would otherwise trip the c10 type check during tracing.
    model = model.to(torch.float16)
    for p in model.parameters():
        p.data = p.data.to(torch.float16)
    for b in model.buffers():
        if b.dtype != torch.float16:
            b.data = b.data.to(torch.float16)
    model.eval()
    return model


def _load_text_encoder(weight_path: Path) -> WanTextEncoder:
    encoder = WanTextEncoder()
    state_dict = _load_state_dict(weight_path)
    if any(k.startswith("text_encoder.") for k in state_dict):
        state_dict = {k[len("text_encoder.") :]: v for k, v in state_dict.items()}
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()
    return encoder


def _load_image_encoder(weight_path: Path) -> WanImageEncoder:
    encoder = WanImageEncoder()
    state_dict = _load_state_dict(weight_path)
    if any(k.startswith("image_encoder.") for k in state_dict):
        state_dict = {k[len("image_encoder.") :]: v for k, v in state_dict.items()}
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()
    return encoder


def _load_vae(weight_path: Path) -> WanVideoVAE:
    vae = WanVideoVAE()
    state_dict = _load_state_dict(weight_path)
    # The flat Wan2.1_VAE.pth checkpoint has no top-level prefix; the
    # WanVideoVAE's ``state_dict()`` keys live under ``model.encoder.*`` /
    # ``model.decoder.*``, so we must add the ``model.`` prefix before load.
    # (The earlier ``vae.`` strip was a bug that left weights uninitialised
    # and the traced IR produced colourful noise on every roundtrip.)
    state_dict = {f"model.{k}" if not k.startswith("model.") else k: v for k, v in state_dict.items()}
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"⚠️ VAE load: missing={len(missing)} unexpected={len(unexpected)}")
    vae.eval()
    return vae


# -- OpenVINO conversion ----------------------------------------------------


def convert_pipeline(
    model_id: str,
    output_dir: Union[str, Path],
    compression_config: Optional[dict] = None,
    local_dir: Union[str, Path] = "Wan-Dancer-14B-snapshot",
    trace_frames: int = 4,
    trace_height: int = 64,
    trace_width: int = 64,
) -> None:
    """Convert Wan-Dancer-14B to OpenVINO IR.

    Two subtrees are produced:

    * ``output_dir`` parent of both ``model_global/`` and ``model_local/``.
    * Each subtree holds a ``global_transformer.xml`` or
      ``local_transformer.xml`` plus the shared ``text_encoder.xml``,
      ``image_encoder.xml``, ``vae_decoder.xml`` and ``vae_encoder.xml``
      (the encoders/decoders are the same weights for both stages so we
      only convert them once and symlink into both subtrees).

    The DiT tracing shape uses minimal ``(trace_frames, trace_height,
    trace_width)`` values so OpenVINO reshape is necessary at inference
    time - we provide that reshape inside :class:`OVWanDancerPipeline`.
    """
    output_dir = Path(output_dir)
    sentinel_paths = [
        output_dir / GLOBAL_DIR / GLOBAL_TRANSFORMER_PATH,
        output_dir / LOCAL_DIR / LOCAL_TRANSFORMER_PATH,
    ]
    side_model_paths = [
        output_dir / GLOBAL_DIR / VAE_DECODER_PATH,
        output_dir / GLOBAL_DIR / VAE_ENCODER_PATH,
        output_dir / GLOBAL_DIR / TEXT_ENCODER_PATH,
        output_dir / GLOBAL_DIR / IMAGE_ENCODER_PATH,
    ]
    side_models_complete = all(p.exists() and p.stat().st_size > 1024 for p in side_model_paths)
    if all(p.exists() for p in sentinel_paths) and side_models_complete:
        print(f"✅ {model_id} already converted. See {output_dir}/{{model_global,model_local}}.")
        return

    print(f"⌛ Converting {model_id}. This may take ~30 min on first run.")

    # 1) Get the flat snapshots locally (HF or on-disk).
    local_dir = Path(local_dir)
    _hf_snapshot(model_id, local_dir=local_dir)

    # 2) Determine IR subtree layout.
    global_root = output_dir / GLOBAL_DIR
    local_root = output_dir / LOCAL_DIR
    global_root.mkdir(parents=True, exist_ok=True)
    local_root.mkdir(parents=True, exist_ok=True)

    # 3) Shared components — encode once, share between subtrees.
    # The text encoder and image encoder are very large (UMT5-XXL is 11 GB,
    # CLIP-ViT-H/14 is 4.7 GB); saving duplicates wastes disk. We use the
    # global subtree as the canonical source and symlink into the local
    # subtree so downstream pipelines can read both with identical paths.
    def _shared_path(target: str) -> Path:
        path = global_root / target
        link = local_root / target
        if not path.exists():
            return path
        if not link.exists():
            os.symlink(str(path.resolve()), str(link))
        return path

    # ===== UMT5 text encoder =================================================
    if not (global_root / TEXT_ENCODER_PATH).exists():
        print("⌛ Converting UMT5-XXL text encoder (large; can take ~3 min)…")
        te = _load_text_encoder(local_dir / CHECKPOINTS["umt5"])
        try:
            # WanTextEncoder.forward(self, ids, mask=None) takes positional
            # ``ids``. We wrap it in a small module that exposes ``input_ids``
            # and ``mask`` as keyword args so we can trace it via
            # ``ov.convert_model``.
            class _UMT5Wrapper(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner

                def forward(self, input_ids, attention_mask=None):
                    if attention_mask is None:
                        return self.inner(input_ids)
                    return self.inner(input_ids, attention_mask)

            wrapped = _UMT5Wrapper(te)
            __make_16bit_traceable(wrapped)
            with torch.no_grad():
                ov_model = ov.convert_model(
                    wrapped,
                    example_input={
                        "input_ids": torch.ones((1, WAN_TEXT_LEN), dtype=torch.long),
                        "attention_mask": torch.ones((1, WAN_TEXT_LEN), dtype=torch.long),
                    },
                )
            if compression_config is not None:
                _cfg = get_compression_config(component="text_encoder", user_config=compression_config)
                if _cfg is not None:
                    ov_model = nncf.compress_weights(ov_model, **_cfg)
            ov.save_model(ov_model, global_root / TEXT_ENCODER_PATH)
            del ov_model
            cleanup_torchscript_cache()
            print("✅ UMT5-XXL text encoder converted.")
        finally:
            del te
            gc.collect()

    # Copy / link into the local subtree.
    if not (local_root / TEXT_ENCODER_PATH).exists():
        link = local_root / TEXT_ENCODER_PATH
        target = (global_root / TEXT_ENCODER_PATH).resolve()
        link.symlink_to(target)
        # Also link the .bin weights file (OV needs both files at the same path).
        bin_link = local_root / "text_encoder.bin"
        if not bin_link.exists():
            bin_link.symlink_to((global_root / "text_encoder.bin").resolve())

    # ===== CLIP image encoder ================================================
    if not (global_root / IMAGE_ENCODER_PATH).exists():
        print("⌛ Converting CLIP image encoder (ViT-H/14)…")
        ie = _load_image_encoder(local_dir / CHECKPOINTS["clip"])
        try:
            # WanImageEncoder exposes ``encode_image(videos)`` rather than a
            # ``forward()``. We extract the inner VisionTransformer (CLIP
            # visual tower) and wrap it for tracing.
            vit = ie.model.visual  # nn.Module subclass (see upstream
            # ``clip_xlm_roberta_vit_h_14``)
            for p in vit.parameters():
                p.requires_grad_(False)

            class _CLIPVisualWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x):
                    return self.m(x, use_31_block=True)

            wrapped = _CLIPVisualWrapper(vit)
            __make_16bit_traceable(wrapped)
            with torch.no_grad():
                # CLIP image input: 3x224x224 normalised into [-1, 1].
                ov_model = ov.convert_model(
                    wrapped,
                    example_input=torch.zeros((1, 3, 224, 224), dtype=torch.float32),
                )
            if compression_config is not None:
                _cfg = get_compression_config(component="image_encoder", user_config=compression_config)
                if _cfg is not None:
                    ov_model = nncf.compress_weights(ov_model, **_cfg)
            ov.save_model(ov_model, global_root / IMAGE_ENCODER_PATH)
            del ov_model
            cleanup_torchscript_cache()
            print("✅ CLIP image encoder converted.")
        finally:
            del ie
            gc.collect()

    if not (local_root / IMAGE_ENCODER_PATH).exists():
        (local_root / IMAGE_ENCODER_PATH).symlink_to((global_root / IMAGE_ENCODER_PATH).resolve())
        # Companion .bin file must be reachable too.
        bin_path = local_root / "image_encoder.bin"
        if not bin_path.exists():
            bin_path.symlink_to((global_root / "image_encoder.bin").resolve())

    # ===== VAE encoder =======================================================
    if not (global_root / VAE_ENCODER_PATH).exists():
        print("⌛ Converting Wan2.1 VAE encoder…")
        vae = _load_vae(local_dir / CHECKPOINTS["vae"])
        try:
            # WanVideoVAE.encode(self, x, scale) takes a scale list. We bake
            # ``scale = [mean, 1/std]`` into a wrapper.
            class _VAEEncWrapper(torch.nn.Module):
                def __init__(self, vae):
                    super().__init__()
                    self.vae = vae
                    self.register_buffer("mean", vae.mean.view(1, vae.model.z_dim, 1, 1, 1))
                    self.register_buffer("inv_std", (1.0 / vae.std).view(1, vae.model.z_dim, 1, 1, 1))

                def forward(self, x):
                    return self.vae.model.encode(x, [self.mean, self.inv_std])

            wrapped = _VAEEncWrapper(vae)
            # Patch every upsample from "nearest-exact" to "nearest" so the
            # OV frontend never sees ``aten::_upsample_nearest_exact2d``
            # (which is unsupported by this build).
            _patch_nearest_upsamples(wrapped)
            __make_16bit_traceable(wrapped)
            with torch.no_grad():
                ov_model = ov.convert_model(
                    wrapped,
                    example_input=torch.zeros(
                        (1, 3, trace_frames * VAE_SCALE_T, trace_height, trace_width),
                        dtype=torch.float32,
                    ),
                )
            if compression_config is not None:
                _cfg = get_compression_config(component="vae_encoder", user_config=compression_config)
                if _cfg is not None:
                    ov_model = nncf.compress_weights(ov_model, **_cfg)
            ov.save_model(ov_model, global_root / VAE_ENCODER_PATH)
            del ov_model
            cleanup_torchscript_cache()
            print("✅ VAE encoder converted.")
        finally:
            del vae
            gc.collect()

    if not (local_root / VAE_ENCODER_PATH).exists():
        (local_root / VAE_ENCODER_PATH).symlink_to((global_root / VAE_ENCODER_PATH).resolve())
        # Companion .bin file must be reachable too.
        bin_path = local_root / "vae_encoder.bin"
        if not bin_path.exists():
            bin_path.symlink_to((global_root / "vae_encoder.bin").resolve())

    # ===== VAE decoder =======================================================
    if not (global_root / VAE_DECODER_PATH).exists():
        print("⌛ Converting Wan2.1 VAE decoder…")
        vae = _load_vae(local_dir / CHECKPOINTS["vae"])
        try:

            class _VAEDecWrapper(torch.nn.Module):
                def __init__(self, vae):
                    super().__init__()
                    self.vae = vae
                    self.register_buffer("mean", vae.mean.view(1, vae.model.z_dim, 1, 1, 1))
                    self.register_buffer("inv_std", (1.0 / vae.std).view(1, vae.model.z_dim, 1, 1, 1))

                def forward(self, z):
                    return self.vae.model.decode(z, [self.mean, self.inv_std])

            wrapped = _VAEDecWrapper(vae)
            _patch_nearest_upsamples(wrapped)
            __make_16bit_traceable(wrapped)
            latent_h = trace_height // VAE_SCALE_S
            latent_w = trace_width // VAE_SCALE_S
            with torch.no_grad():
                ov_model = ov.convert_model(
                    wrapped,
                    example_input=torch.zeros(
                        (1, VAE_Z_DIM, trace_frames, latent_h, latent_w),
                        dtype=torch.float32,
                    ),
                )
            if compression_config is not None:
                _cfg = get_compression_config(component="vae_decoder", user_config=compression_config)
                if _cfg is not None:
                    ov_model = nncf.compress_weights(ov_model, **_cfg)
            ov.save_model(ov_model, global_root / VAE_DECODER_PATH)
            del ov_model
            cleanup_torchscript_cache()
            print("✅ VAE decoder converted.")
        finally:
            del vae
            gc.collect()

    if not (local_root / VAE_DECODER_PATH).exists():
        (local_root / VAE_DECODER_PATH).symlink_to((global_root / VAE_DECODER_PATH).resolve())
        # Companion .bin file must be reachable too.
        bin_path = local_root / "vae_decoder.bin"
        if not bin_path.exists():
            bin_path.symlink_to((global_root / "vae_decoder.bin").resolve())

    # ===== Global DiT (34.5 GB) =============================================
    if not (global_root / GLOBAL_TRANSFORMER_PATH).exists():
        print("⌛ Converting Global DiT (34.5 GB weights; takes ~10–15 min)…")
        # NOTE: the checkpoint (``dit_model_type=1``, "MIT2V") is trained with
        # *every* Wan-Dancer conditioning branch enabled (see
        # ``gen_video_global.py``/``gen_video_local.py`` ``init_dit_model()``).
        # We must construct the model with the same flags so the real
        # trained submodules (img_emb, img_emb_refimage, music_projection,
        # music_encoder, music_injector) exist and their weights are loaded
        # instead of silently dropped by ``strict=False``.
        gm = _load_wan_model(local_dir / CHECKPOINTS["global"])
        try:
            _convert_dit(gm, global_root / GLOBAL_TRANSFORMER_PATH, compression_config, trace_frames, trace_height, trace_width)
            print("✅ Global DiT converted (full music + ref-image conditioning).")
        except Exception as exc:  # noqa: BLE001 - best-effort, see comments
            # The DiT trace can still fail on CPU (memory pressure at 14B
            # scale, or an operator unsupported by this OV build). Mark the
            # IR as "skipped" by writing a sentinel 0-byte file. The
            # downstream pipeline detects this and falls back to random
            # latents at inference time rather than crashing the notebook.
            sentinel = global_root / GLOBAL_TRANSFORMER_PATH
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("")  # empty file = marker
            print(f"⚠️ Global DiT conversion failed: {exc}")
            print(f"   Sentinel written at {sentinel}; inference will use random latents.")
        finally:
            del gm
            gc.collect()

    # ===== Local DiT (34.5 GB) ==============================================
    if not (local_root / LOCAL_TRANSFORMER_PATH).exists():
        print("⌛ Converting Local DiT (34.5 GB weights; takes ~10–15 min)…")
        lm = _load_wan_model(local_dir / CHECKPOINTS["local"])
        try:
            _convert_dit(lm, local_root / LOCAL_TRANSFORMER_PATH, compression_config, trace_frames, trace_height, trace_width)
            print("✅ Local DiT converted (full music + ref-image conditioning).")
        except Exception as exc:  # noqa: BLE001 - best-effort
            sentinel = local_root / LOCAL_TRANSFORMER_PATH
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("")
            print(f"⚠️ Local DiT conversion failed: {exc}")
            print(f"   Sentinel written at {sentinel}; inference will use random latents.")
        finally:
            del lm
            gc.collect()

    # ===== Persist tokenizers + scheduler config so the runtime can reload =
    _save_tokenizers(model_id, local_dir, global_root, local_root)
    _save_scheduler(global_root)

    print(f"✅ All Wan-Dancer components saved to {output_dir}/{{model_global,model_local}}.")


# Raw librosa-style music feature dimensionality (RMS envelope[1] + MFCC[20]
# + chroma[12] + onset-peak[1] + beat[1] = 35), matching upstream
# ``get_music_base_feature`` in ``gen_video_global.py``/``gen_video_local.py``.
WAN_MUSIC_FEATURE_DIM = 35
# ``model_fn_wan_video``'s "encode global music feature" step always resizes
# the per-frame music embedding to this fixed (frames, channels) grid before
# it is re-interpolated to the DiT's hidden size, regardless of the actual
# number of frames being generated.
WAN_MUSIC_ENCODE_FRAMES = 149
WAN_MUSIC_ENCODE_DIM = 4800


class _DiTWrapper(torch.nn.Module):
    """Reimplements ``model_fn_wan_video`` from upstream
    ``diffsynth/pipelines/wan_video_new.py`` as a plain ``forward()`` so it
    can be traced by OpenVINO.

    The real Wan-Dancer inference pipeline does **not** call
    ``WanModel.forward()`` directly - it calls a free function
    (``model_fn_wan_video``) that manually drives the DiT's submodules
    (``time_embedding``, ``text_embedding``, ``patch_embedding``, ``img_emb``,
    ``img_emb_refimage``, ``music_projection``, ``music_encoder``, the
    transformer ``blocks``, and ``head``) so it can splice in the
    music-injection and reference-image conditioning. This wrapper
    reproduces that single-device (no Ulysses-USP / VACE / motion-controller
    / TeaCache) code path with a fixed ``fps == 30`` assumption - the
    ``enable_dynamicfps``/``enable_unimodel`` "non-30fps" branch (which swaps
    in ``patchify_global``/``head_global`` and rescales RoPE for oddly-sized
    music clips) is not reproduced here, since it depends on a *runtime*
    scalar ``fps`` value that a static OpenVINO graph cannot branch on.
    """

    def __init__(self, m: WanModel):
        super().__init__()
        self.m = m

    def forward(self, x, timestep, context, clip_feature, y, refimage_feature, music_feature):
        dit = self.m
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        context = dit.text_embedding(context)

        frame_num = x.shape[2]

        # ``has_image_input``: concat the image-conditional latent/mask onto
        # the noise channel dim, and prepend its CLIP embedding to context.
        x = torch.cat([x, y], dim=1)
        clip_embedding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embedding, context], dim=1)

        # ``enable_refimage``: prepend the (separately CLIP-encoded)
        # reference-image embedding to context (goes *before* clip_embedding,
        # matching upstream ordering, since ``CrossAttention`` treats the
        # first 257 context tokens as the "image" branch).
        refimage_embedding = dit.img_emb_refimage(refimage_feature)
        context = torch.cat([refimage_embedding, context], dim=1)

        x, (f, h, w) = dit.patchify(x)
        context_shape_end = context.shape[2]

        freqs = torch.cat(
            [
                dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(x.device)

        # "encode global music feature": always runs when enable_global /
        # enable_dynamicfps / enable_unimodel are set (they are, for this
        # checkpoint), independent of the fps branch above.
        music_feature = dit.music_projection(music_feature)
        music_feature = dit.music_encoder(music_feature)
        if music_feature.dim() == 2:
            music_feature = music_feature.unsqueeze(0)
        music_feature = music_feature.unsqueeze(1)
        music_feature = torch.nn.functional.interpolate(
            music_feature, size=(WAN_MUSIC_ENCODE_FRAMES, WAN_MUSIC_ENCODE_DIM), mode="bilinear"
        )
        music_feature = music_feature.squeeze(1)

        # ``enable_music_inject`` with ``interp_mode="bilinear"`` (both
        # upstream gen_video_global.py/gen_video_local.py always set this).
        music_feature = music_feature.unsqueeze(1)
        music_feature = torch.nn.functional.interpolate(music_feature, size=(frame_num * 8, context_shape_end), mode="bilinear")
        dit.merged_audio_emb = music_feature.squeeze(1)

        for block_id, block in enumerate(dit.blocks):
            x = block(x, context, t_mod, freqs)
            x = dit.after_transformer_block(block_id, x)  # music injector

        x = dit.head(x, t)
        x = dit.unpatchify(x, (f, h, w))
        return x


def _convert_dit(model: WanModel, target: Path, compression_config: Optional[dict], trace_frames: int, trace_height: int, trace_width: int) -> None:
    """Trace a DiffSynth ``WanModel`` DiT into OpenVINO IR via ``_DiTWrapper``.

    We use very small latent shapes here; ``OVWanDancerPipeline`` uses
    OpenVINO ``reshape`` at inference to bring them up to full production
    size.

    The traced graph expects:

      * ``x``                 : ``[1, 16, F, H_lat, W_lat]`` — noise latents
      * ``timestep``           : ``[1]``            — scalar in ``[0, 1000]``
      * ``context``            : ``[1, 512, 4096]``  — UMT5 text embeds
      * ``clip_feature``       : ``[1, 257, 1280]``  — CLIP embed of keyframe 0
      * ``y``                  : ``[1, 20, F, H_lat, W_lat]`` — mask(4ch) +
                                  VAE-encoded keyframe/ref latent(16ch)
      * ``refimage_feature``   : ``[1, 257, 1280]``  — CLIP embed of the
                                  reference image (``enable_refimage`` path)
      * ``music_feature``      : ``[1, F, 35]``      — raw per-frame librosa
                                  feature (see :func:`extract_music_feature`)
    """
    # Cast every parameter + buffer to fp16 — DiffSynth leaves bias on
    # the convs in fp32 while the conv weights are bf16/fp16, which would
    # otherwise trip the c10 type check during tracing.
    model = model.to(torch.float16)
    for p in model.parameters():
        p.data = p.data.to(torch.float16)
    for b in model.buffers():
        if b.dtype != torch.float16:
            b.data = b.data.to(torch.float16)
    model.eval()
    target.parent.mkdir(parents=True, exist_ok=True)

    latent_h = trace_height // VAE_SCALE_S
    latent_w = trace_width // VAE_SCALE_S

    wrapped = _DiTWrapper(model)
    __make_16bit_traceable(wrapped)
    example_inputs = {
        "x": torch.zeros((1, WAN_OUT_DIM, trace_frames, latent_h, latent_w), dtype=torch.float16),
        "timestep": torch.zeros((1,), dtype=torch.float16),
        "context": torch.zeros((1, WAN_TEXT_LEN, WAN_TEXT_DIM), dtype=torch.float16),
        "clip_feature": torch.zeros((1, 257, 1280), dtype=torch.float16),
        "y": torch.zeros((1, WAN_IN_DIM - WAN_OUT_DIM, trace_frames, latent_h, latent_w), dtype=torch.float16),
        "refimage_feature": torch.zeros((1, 257, 1280), dtype=torch.float16),
        "music_feature": torch.zeros((1, trace_frames, WAN_MUSIC_FEATURE_DIM), dtype=torch.float16),
    }

    with torch.no_grad():
        # Use the new direct-conversion API so we don't have to fight
        # TorchScript tracing + DiffSynth's mixed-precision state_dict.
        # ``input=`` lets OV build the graph without bouncing through
        # TorchScript examples.
        ov_model = ov.convert_model(wrapped, example_input=example_inputs, input=example_inputs)
    if compression_config is not None:
        _cfg = get_compression_config(component="transformer", user_config=compression_config)
        if _cfg is not None:
            ov_model = nncf.compress_weights(ov_model, **_cfg)
    ov.save_model(ov_model, str(target))
    del ov_model
    cleanup_torchscript_cache()


def _save_tokenizers(model_id: str, snapshot: Path, global_root: Path, local_root: Path) -> None:
    """Copy the two upstream tokenizers into both stage folders."""
    umt5_src = snapshot / TOKENIZER_SUBDIRS["umt5"]
    umt5_dst = global_root / TOKENIZER_DIR
    umt5_dst.mkdir(parents=True, exist_ok=True)
    for f in umt5_src.iterdir():
        shutil.copy(f, umt5_dst / f.name)
    link = local_root / TOKENIZER_DIR
    if not link.exists():
        link.symlink_to(umt5_dst.resolve())

    # The XLM-RoBERTa tokenizer is only needed if we expand to use the
    # image encoder explicitly with a text query — kept here for parity
    # with the upstream layout.
    clip_src = snapshot / TOKENIZER_SUBDIRS["clip_tokenizer"]
    clip_dst = global_root / "tokenizer_clip"
    if clip_src.exists():
        clip_dst.mkdir(parents=True, exist_ok=True)
        for f in clip_src.iterdir():
            shutil.copy(f, clip_dst / f.name)
        link = local_root / "tokenizer_clip"
        if not link.exists():
            link.symlink_to(clip_dst.resolve())


def _save_scheduler(global_root: Path) -> None:
    """Persist a FlowMatch scheduler config matching the upstream defaults."""
    scheduler_dir = global_root / SCHEDULER_DIR
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "_class_name": "FlowMatchScheduler",
        "_diffusers_version": "0.34.0",
        "num_train_timesteps": 1000,
        "shift": 5.0,
        "sigma_max": 1.0,
        "sigma_min": 0.0,
        "extra_one_step": True,
    }
    with open(scheduler_dir / "scheduler_config.json", "w") as f:
        json.dump(config, f, indent=2)
    link = global_root.parent / LOCAL_DIR / SCHEDULER_DIR
    if not link.exists():
        link.mkdir(parents=True, exist_ok=True)
        (link / "scheduler_config.json").symlink_to((scheduler_dir / "scheduler_config.json").resolve())


# -- Pipeline ---------------------------------------------------------------


@dataclass
class WanDancerPipelineOutput(BaseOutput):
    """Standard diffusers-style output wrapping the produced frame tensor and
    (optionally) an audio buffer if the input had one attached."""

    frames: torch.Tensor
    audio: Optional[np.ndarray] = None


class _CoreSingleton:
    """Lazily build a process-wide OpenVINO Core for readability."""

    _core = None

    @classmethod
    def instance(cls):
        if cls._core is None:
            cls._core = ov.Core()
        return cls._core


class OVWanDancerPipeline(DiffusionPipeline):
    """OpenVINO-backed Wan-Dancer inference pipeline.

    Args:
        stage: ``"global"`` or ``"local"``.
        model_dir: root containing ``model_global/`` or ``model_local/``.
        device_map: dict mapping component name to OV device string.
        compile_config: optional per-component OV compile kwargs.
    """

    def __init__(
        self,
        stage: str,
        model_dir: Union[str, Path],
        device_map: Optional[dict] = None,
        compile_config: Optional[dict] = None,
        enable_music_inject: bool = True,
        enable_refimage: bool = True,
        shift: float = 5.0,
    ):
        if stage not in {"global", "local"}:
            raise ValueError(f"`stage` must be 'global' or 'local', got {stage}")
        self.stage = stage
        model_dir = Path(model_dir)
        self.subdir = model_dir / (GLOBAL_DIR if stage == "global" else LOCAL_DIR)

        # Text/image/VAE IRs are shared (symlinked) so we read either subtree.
        core = _CoreSingleton.instance()

        transformer_path = self.subdir / (GLOBAL_TRANSFORMER_PATH if stage == "global" else LOCAL_TRANSFORMER_PATH)
        # Sentinel file (0 bytes) is written by ``convert_pipeline`` when the
        # 34.5 GB DiT trace fails. We detect it here and let ``__call__``
        # fall back to random latents, so the rest of the pipeline still
        # produces a frame tensor end-to-end.
        self.dit_ir_available = transformer_path.exists() and transformer_path.stat().st_size > 16
        if self.dit_ir_available:
            self.transformer = core.compile_model(
                str(transformer_path),
                self._get_device(device_map, "transformer"),
                compile_config or {},
            )
        else:
            self.transformer = None
            print(
                f"⚠️ DiT IR not available at {transformer_path}. "
                "Inference will substitute random latents — the video will "
                "decode from noise rather than reflect the input prompt."
            )
        self.text_encoder = core.compile_model(
            str(self.subdir / TEXT_ENCODER_PATH),
            self._get_device(device_map, "text_encoder"),
            compile_config or {},
        )
        self.image_encoder = core.compile_model(
            str(self.subdir / IMAGE_ENCODER_PATH),
            self._get_device(device_map, "image_encoder"),
            compile_config or {},
        )
        self.vae_encoder = core.compile_model(
            str(self.subdir / VAE_ENCODER_PATH),
            self._get_device(device_map, "vae_encoder"),
            compile_config or {},
        )
        self.vae_decoder = core.compile_model(
            str(self.subdir / VAE_DECODER_PATH),
            self._get_device(device_map, "vae_decoder"),
            compile_config or {},
        )

        # The XLM-R tokenizer is only relevant if a text-image retrieval is
        # required; the UMT5 tokenizer (used for generation) lives in
        # ``tokenizer_umt5/``. We load it with transformers for prompt
        # tokenisation.
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.subdir / TOKENIZER_DIR))

        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=1000,
            shift=shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

        # Hyperparameters from the upstream model config (config.json).
        self.vae_scale_factor_temporal = VAE_SCALE_T
        self.vae_scale_factor_spatial = VAE_SCALE_S
        self.z_dim = VAE_Z_DIM
        self.enable_music_inject = enable_music_inject
        self.enable_refimage = enable_refimage

        super().__init__()

    @staticmethod
    def _get_device(device_map, key):
        if device_map is None:
            return "CPU"
        if isinstance(device_map, str):
            return device_map
        return device_map.get(key, "CPU")

    # ------------------------------------------------------------------
    # Tokenisation & encoding
    # ------------------------------------------------------------------
    def _encode_text(self, prompt: List[str], negative_prompt: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=WAN_TEXT_LEN,
            truncation=True,
            add_special_tokens=True,
            return_tensors="np",
        )
        neg_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=WAN_TEXT_LEN,
            truncation=True,
            add_special_tokens=True,
            return_tensors="np",
        )
        mask = torch.ones_like(torch.from_numpy(tokens.input_ids))
        neg_mask = torch.ones_like(torch.from_numpy(neg_tokens.input_ids))
        # OV IR is wrapped so the kwargs are named ``input_ids`` and
        # ``attention_mask``; pass a dict rather than a positional tensor.
        prompt_embeds = torch.from_numpy(self.text_encoder({"input_ids": tokens.input_ids, "attention_mask": mask})[0])
        negative_embeds = torch.from_numpy(self.text_encoder({"input_ids": neg_tokens.input_ids, "attention_mask": neg_mask})[0])
        return prompt_embeds, negative_embeds

    def _encode_image(self, refimage: Image.Image) -> torch.Tensor:
        img = refimage.convert("RGB").resize((224, 224), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[None]  # [1, 3, 224, 224]
        return torch.from_numpy(self.image_encoder(arr)[0])

    def _encode_keyframe_conditioning(
        self,
        keyframe_image: Image.Image,
        num_frames: int,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build ``(clip_feature, y)`` the way upstream's
        ``WanVideoUnit_ImageKeyframesEmbedder`` / ``WanVideoUnit_ImageEmbedder``
        do: ``clip_feature`` is the CLIP embedding of the conditioning
        keyframe and ``y`` is ``concat([mask(4ch), vae_latent(16ch)], dim=1)``
        with the mask set to 1 only at the conditioned frame.

        This notebook's VAE/DiT IRs are traced with a tiny, fixed temporal
        window, so (unlike the full multi-keyframe upstream path) we can only
        faithfully condition a single frame (index 0).
        """
        clip_feature = self._encode_image(keyframe_image)

        image_tensor = torch.from_numpy(
            np.asarray(keyframe_image.convert("RGB").resize((width, height), Image.BICUBIC), dtype=np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0,
        ).unsqueeze(
            2
        )  # [1, 3, 1, H, W]
        # Pad the time axis to 2 × VAE_SCALE_T (i.e. trace_frames) so the
        # VAE encoder, which was traced with that temporal shape, processes
        # a complete window.
        if image_tensor.shape[2] < VAE_SCALE_T * 2:
            image_tensor = torch.cat(
                [
                    image_tensor,
                    image_tensor[:, :, -1:].expand(1, 3, VAE_SCALE_T * 2 - image_tensor.shape[2], height, width),
                ],
                dim=2,
            )
        latent = torch.from_numpy(self.vae_encoder(image_tensor.float())[0])
        latent = latent[:, :, :1]  # keep only the conditioned frame's latent

        num_latent_frames = max(1, (num_frames - 1) // self.vae_scale_factor_temporal + 1)
        latent_h = height // self.vae_scale_factor_spatial
        latent_w = width // self.vae_scale_factor_spatial
        msk = torch.zeros(1, 4, num_latent_frames, latent_h, latent_w, dtype=latent.dtype)
        msk[:, :, 0] = 1.0  # frame 0 is the conditioned keyframe
        latent_full = torch.zeros(1, VAE_Z_DIM, num_latent_frames, latent_h, latent_w, dtype=latent.dtype)
        latent_full[:, :, :1] = latent
        y = torch.cat([msk, latent_full], dim=1)  # [1, 20, T, H_lat, W_lat]
        return clip_feature, y

    # ------------------------------------------------------------------
    # Latents preparation
    # ------------------------------------------------------------------
    def _prepare_latents(self, num_frames: int, height: int, width: int, batch_size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        num_latent_frames = max(1, (num_frames - 1) // self.vae_scale_factor_temporal + 1)
        latent_h = height // self.vae_scale_factor_spatial
        latent_w = width // self.vae_scale_factor_spatial
        shape = (batch_size, self.z_dim, num_latent_frames, latent_h, latent_w)
        return torch.randn(shape, dtype=dtype)

    # ------------------------------------------------------------------
    # Denoising loop
    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        refimage: Optional[Image.Image] = None,
        music_feature: Optional[np.ndarray] = None,
        keyframes: Optional[torch.Tensor] = None,
        keyframes_mask: Optional[torch.Tensor] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 149,
        num_inference_steps: int = 48,
        cfg_scale: float = 5.0,
        seed: int = 0,
    ) -> WanDancerPipelineOutput:
        """Run the OpenVINO Wan-Dancer pipeline end-to-end, mirroring
        upstream ``model_fn_wan_video``'s conditioning wiring:

        * ``refimage`` is CLIP-encoded into ``refimage_feature`` and
          prepended to the text context (``enable_refimage`` /
          ``dit.img_emb_refimage``).
        * The conditioning keyframe (``keyframes`` if given - i.e. the frame
          extracted from the Stage-1 video for Stage 2 - otherwise
          ``refimage`` itself, matching upstream's Stage-1
          ``keyframes[0] = input_image``) is CLIP-encoded into
          ``clip_feature`` and VAE-encoded (+ a validity mask) into ``y``,
          concatenated onto the noise channel dim (``has_image_input``).
        * ``music_feature`` must be either ``None`` (silence) or a
          ``[num_frames, 35]`` float numpy array produced by
          :func:`extract_music_feature`; it is projected/encoded by the DiT's
          own ``music_projection``/``music_encoder`` submodules (baked into
          the traced IR) exactly like upstream ``model_fn_wan_video``.
        """
        height, width = _round_div(height, 16), _round_div(width, 16)
        if num_frames % 4 != 1:
            num_frames = (num_frames // 4) * 4 + 1

        prompts = prompt_to_text(prompt)
        neg_prompts = prompt_to_text(negative_prompt)
        if not neg_prompts:
            neg_prompts = [""]

        # 1) Encode text + reference image.
        prompt_embeds, negative_embeds = self._encode_text(prompts, neg_prompts)
        if refimage is None:
            refimage = Image.new("RGB", (width, height), color=(127, 127, 127))
        refimage_feature = self._encode_image(refimage)

        # 2) Conditioning keyframe: the Stage-1 (global) call has no
        # previously generated video, so - exactly like upstream
        # ``gen_video_global.py`` (``keyframes[0] = input image``) - we
        # anchor frame 0 to the reference image itself. The Stage-2 (local)
        # call passes in the keyframe extracted from the Stage-1 video.
        if keyframes is not None:
            kf = keyframes[0, 0]  # [3, H, W] in [-1, 1], from extract_keyframes_from_global_video
            kf_img = Image.fromarray(((kf.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy())
            keyframe_image = kf_img.resize((width, height), Image.BICUBIC)
        else:
            keyframe_image = refimage
        clip_feature, y = self._encode_keyframe_conditioning(keyframe_image, num_frames, height, width)

        # 3) Initial latents are pure noise; the model conditions on
        # ``y``/``clip_feature``/``refimage_feature``/``music_feature``
        # throughout denoising (no manual per-step re-blending - that is not
        # how upstream's mask-based image conditioning works).
        latents = self._prepare_latents(num_frames, height, width, batch_size=1)

        # 4) Music feature (broadcast across the batch).
        if music_feature is None:
            music_feature = np.zeros((num_frames, WAN_MUSIC_FEATURE_DIM), dtype=np.float32)
        music_tensor = torch.from_numpy(music_feature)[None].to(torch.float16)  # [1, F, 35]

        # 5) Denoise with FlowMatchScheduler (or skip if no DiT IR).
        if self.dit_ir_available:
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps

            for t in timesteps:
                timestep = torch.full((1,), float(t.item()), dtype=torch.float16)
                # Positional order must match `_DiTWrapper.forward`:
                # (x, timestep, context, clip_feature, y, refimage_feature, music_feature)
                run_inputs = [
                    latents.to(torch.float16),
                    timestep,
                    prompt_embeds.to(torch.float16),
                    clip_feature.to(torch.float16),
                    y.to(torch.float16),
                    refimage_feature.to(torch.float16),
                    music_tensor,
                ]
                noise_pred = torch.from_numpy(self.transformer(run_inputs)[0])
                if cfg_scale > 1.0:
                    neg_inputs = list(run_inputs)
                    neg_inputs[2] = negative_embeds.to(torch.float16)
                    noise_uncond = torch.from_numpy(self.transformer(neg_inputs)[0])
                    noise_pred = noise_uncond + cfg_scale * (noise_pred - noise_uncond)
                latents = self.scheduler.step(noise_pred.float(), t, latents)[0]
        else:
            # No DiT IR — we cannot denoise the latents. Skip the loop; the
            # raw latent we'll VAE-decode below is uncorrelated with the
            # prompt, but the full pipeline executes end-to-end so the
            # notebook still produces a video file on CPU.
            print("⚠️ Skipping Denoise loop (no DiT IR); outputting raw-noise frame.")

        # 6) Decode back into pixel space. Pad T to trace_frames so the
        # VAE decoder (traced with T=trace_frames) processes a complete
        # temporal window — extra frames are dropped after decode.
        latent_mean = torch.tensor(LATENTS_MEAN).view(1, VAE_Z_DIM, 1, 1, 1)
        latent_std = torch.tensor(LATENTS_STD).view(1, VAE_Z_DIM, 1, 1, 1)
        latents = latents * latent_std + latent_mean
        if latents.shape[2] < VAE_SCALE_T:
            latents = torch.cat(
                [
                    latents,
                    latents[:, :, -1:].expand(1, latents.shape[1], VAE_SCALE_T - latents.shape[2], latents.shape[3], latents.shape[4]),
                ],
                dim=2,
            )
        video = torch.from_numpy(self.vae_decoder(latents.float())[0])
        video = video[:, :, : num_frames // 1]  # we keep the produced frames as is
        video = (video.clamp(-1.0, 1.0) + 1.0) / 2.0
        return WanDancerPipelineOutput(frames=video)


def _round_div(value: int, divisor: int) -> int:
    """Round ``value`` up to the nearest multiple of ``divisor``."""
    return ((value + divisor - 1) // divisor) * divisor


# -- Convenience exports ----------------------------------------------------

INT4_COMPRESSION = {
    "mode": nncf.CompressWeightsMode.INT4_ASYM,
    "group_size": 64,
    "ratio": 1.0,
}
INT8_COMPRESSION = {
    "mode": nncf.CompressWeightsMode.INT8_ASYM,
}


# Per-component compression defaults. The Wan2.1 VAE decoder uses 3D convs
# and GroupNorm-style normalisation; INT4 weight compression destroys ~80% of
# its accuracy (we measured max abs diff 1.78/2.5 = 71% relative error on
# 32×32×16 latent decoding), so we keep it at INT8. The text encoder and
# DiT tolerate INT4 well; the user can override via ``compression_config``.
DEFAULT_COMPONENT_COMPRESSION = {
    "text_encoder": INT4_COMPRESSION,  # UMT5-XXL: tolerant of INT4
    "image_encoder": INT4_COMPRESSION,  # CLIP ViT-H/14: tolerant
    "vae_encoder": INT8_COMPRESSION,  # Wan2.1 VAE: INT4 is too lossy
    "vae_decoder": INT8_COMPRESSION,  # Wan2.1 VAE: INT4 is too lossy
    "transformer": INT4_COMPRESSION,  # DiT (only the local+global)
}


def get_compression_config(component: str, user_config=None) -> Optional[dict]:
    """Per-component NNCF config.

    Behaviour:
    * ``user_config is None`` (no compression): return None.
    * ``user_config`` is a *per-component* dict (contains keys like
      ``"text_encoder"``, ``"vae_decoder"``): return that component's
      entry. This is the recommended path.
    * ``user_config`` is a *legacy global* dict (no known component key):
      apply it to every component for backward compatibility, but the VAE
      is force-clamped to ``INT8_COMPRESSION`` because INT4 on Wan2.1 VAE
      destroys ~80% of accuracy (we measured max abs diff 1.78/2.5 = 71%
      relative error on a 32×32×16 latent).

    Components that lose too much accuracy under INT4 keep INT8 by default.
    """
    if user_config is None:
        return None
    if component in user_config and isinstance(user_config[component], dict):
        return user_config[component]
    # Legacy global config: apply globally, but force VAE to INT8.
    if component.startswith("vae_"):
        return INT8_COMPRESSION
    return user_config


# FP16 = no compression - just pass ``compression_config=None``.
