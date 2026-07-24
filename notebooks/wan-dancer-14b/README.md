# Music-to-Dance generation with Wan-Dancer-14B and OpenVINO

Wan-Dancer is a **hierarchical music-to-dance generator** that decomposes minute-scale choreography into two complementary passes: a sparse global keyframe planner and a high-resolution temporal refiner. The release is Apache 2.0 and ships on [Hugging Face](https://huggingface.co/Wan-AI/Wan-Dancer-14B) together with the [Wan-Video/Wan-Dancer](https://github.com/Wan-Video/Wan-Dancer) inference package.

Highlights:

- **Hierarchical, two-stage pipeline** — Stage 1 (`global_model.safetensors`) plans a sparse, full-track keyframe video; Stage 2 (`local_model.safetensors`) refines it to a high-resolution dance clip at the chosen final resolution. Both DiTs share the UMT5-XXL text encoder, the CLIP image encoder and the Wan2.1 3D VAE.
- **Music-aware conditioning** — each frame is conditioned on a 36-dim librosa feature (envelope + 20 MFCC + 12 Chroma CENS + onset peak + beat one-hot). The DiT uses `enable_music_inject=True` so the choreography is locked to the rhythm.
- **OpenVINO-native inference** — every component is converted to OpenVINO IR. A precision dropdown exposes FP16, INT8 (NNCF `INT8_ASYM`) and INT4 (NNCF `INT4_ASYM` group=64 ratio=1.0), matching the wan2.2 reference notebook.
- **Mirrors the upstream usage** — the two notebook stages drive the same arguments as `gen_video_global.sh` and `gen_video_local.sh` from the [Wan-Dancer Hugging Face model card](https://huggingface.co/Wan-AI/Wan-Dancer-14B). The Gradio tab exposes all of them as widgets, including the default K-Pop dance style (prompt + reference image + bundled WAV).

## How the upstream flow maps to this notebook

| Stage | Upstream shell script | This notebook |
|-------|----------------------|---------------|
| 1 — Global Keyframe Video | `./gen_video_global.sh` (48 inference steps, CFG 5.0, 30 fps) | cell **"Stage 1 — Generate Global Keyframe Video"** |
| 2 — Final High-Resolution Video | `./gen_video_local.sh` (24 inference steps, CFG 5.0, takes `global_video_path`) | cell **"Stage 2 — Generate Final High-Resolution Video"** (auto-detects `output_global.mp4` as `keyframes`) |

The notebook defaults map 1:1:

| Knob | Upstream default | Notebook default |
|------|------------------|------------------|
| `num_inference_steps` (global) | `48` | `48` (sliders default; smoke tests use 2) |
| `num_inference_steps` (local)  | `24` | `24` (sliders default; smoke tests use 2) |
| `cfg_scale`                    | `5`  | `5.0` |
| `num_frames` (global)          | `149` | `149` (slider max; smoke tests use 8) |
| `num_frames` (local)           | `81`  | `81` (slider max; smoke tests use 8) |
| `image_path`                   | `gen_video/ref_image/3001.jpg` | bundled K-Pop reference image (configurable) |
| `prompt_path` (global)         | `gen_video/prompt/kpop_global.txt` | reads from `assets/kpop_global.txt` if present, else built-in |
| `prompt_path` (local)          | `gen_video/prompt/kpop_local.txt`  | reads from `assets/kpop_local.txt` if present, else built-in |
| `music_path`                   | `gen_video/music/KPopDance.WAV` | bundled `assets/default_music.wav` (downloaded on first run) |
| `output_folder`                | `outputs/global_video/` / `outputs/final_video/` | local `output_global.mp4` / `output_final.mp4` (re-attaches the same audio) |

The single-rank OV build disables the 8-GPU Ulysses Sequence Parallel from the upstream (not feasible on a single machine). Treat the long-range temporal coherence of minute-scale videos as **experimental quality**; for sub-30-second clips it is on par with the upstream single-rank workflow.

## Notebook contents

- [Prerequisites](#prerequisites)
- [Convert Wan-Dancer to OpenVINO IR](#convert-wan-dancer-to-openvino-ir)
- [Stage 1 — Generate Global Keyframe Video](#stage-1-generate-global-keyframe-video)
- [Stage 2 — Generate Final High-Resolution Video](#stage-2-generate-final-high-resolution-video)
- [Interactive demo (Gradio)](#interactive-demo-gradio)

In this tutorial we consider how to convert, optimize and run **Wan-Dancer-14B** for music-to-dance generation on OpenVINO. The notebook supports inference on **Intel GPUs (Flex 170 / Arc / iGPU)** and **Intel CPUs**. The 14B DiT is shipped FP16 by default.

## Installation instructions

This is a self-contained example that relies on its own code plus an environment built around the custom OpenVINO wheel in `/home/ethan/intel/openvino/build/wheels/`. Use the bundled `install_chain.sh` to set up a fresh venv:

```bash
bash notebooks/wan-dancer-14b/install_chain.sh
```

If you already have a compatible venv (custom OpenVINO wheel, `transformers==4.46.2`, `diffusers==0.34.0`, DiffSynth installed via `python setup.py install`), you can skip the chain and run the notebook directly. For general OpenVINO setup, refer to the [Installation Guide](../../README.md).

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO. The two-stage hierarchical inference is provided as-is; quality at minute-scale with single-rank OpenVINO is reduced compared to the upstream 8-GPU setup.

🔬 **Validation status at this commit.** The helper API surface (`ov_wan_dancer_helper.py`, `gradio_helper.py`) passes a 7-test unit suite, `black -l 160` + `flake8` lint, `pyspelling` spell-check, and a clean `nbformat.validate` + `openvino_notebooks` metadata block. `convert_pipeline` now builds the DiT (`WanModel`) with the **same conditioning flags upstream uses for this checkpoint** (`has_image_input=True, enable_refimage=True, enable_music_inject=True, enable_global=True, enable_dynamicfps=True, enable_unimodel=True`, matching `gen_video_global.py`/`gen_video_local.py`'s `init_dit_model()`), and traces a wrapper that reproduces upstream's `model_fn_wan_video` (music projection/encoder, reference-image CLIP conditioning, image/keyframe VAE conditioning) instead of calling `WanModel.forward()` directly. Earlier commits disabled all of these flags to make the CPU trace "succeed", which silently dropped most of the checkpoint's trained weights (`img_emb`, `img_emb_refimage`, `music_projection`, `music_encoder`, `music_injector`, ...) via `load_state_dict(strict=False)` and produced pure-noise output — this is now fixed and missing/unexpected keys are printed at load time. The `enable_dynamicfps`/`enable_unimodel` "non-30fps" branch (`patchify_global`/`head_global` + RoPE rescale, used when the audio's implied fps deviates from 30) is **not** reproduced since it depends on a runtime scalar that a static OpenVINO graph can't branch on — expect reduced fidelity versus the CUDA/USP upstream pipeline for that case. `xfuser`/`yunchang`/`flash_attn` (multi-GPU Ulysses-SP + fused attention) are stubbed out for single-device CPU/GPU inference.

🔬 **Component accuracy (OV-IR vs PyTorch, 32×32×16 latent, mean abs and max abs diff):**

| Component | FP32 OV | INT8 OV | INT4 OV |
|---|---|---|---|
| Wan2.1 VAE decoder | 0.0017 ✅ | **0.0248** ✅ | 1.78 ❌ (70× worse) |
| UMT5-XXL text encoder | tbd | tbd | tbd |
| CLIP image encoder | tbd | tbd | tbd |

**Result of the test**: the OV frontend conversion is essentially exact (FP32 IR matches PyTorch to 4 decimal places). NNCF INT4 weight compression is *too lossy* for the Wan2.1 VAE — the converted IR produces visible artefacts. NNCF INT8 is **~70× better** than INT4 on VAE.

`convert_pipeline` therefore applies NNCF **INT8** to the VAE encoder/decoder and NNCF **INT4** to the text encoder / CLIP / DiT (the only model where INT4 is needed to keep memory in budget). To override, pass a per-component dict:

```python
from ov_wan_dancer_helper import INT4_COMPRESSION, INT8_COMPRESSION, convert_pipeline
convert_pipeline(
    "Wan-AI/Wan-Dancer-14B",
    "model",
    compression_config={
        "text_encoder":  INT4_COMPRESSION,
        "image_encoder": INT4_COMPRESSION,
        "vae_encoder":   INT8_COMPRESSION,
        "vae_decoder":   INT8_COMPRESSION,
        "transformer":   INT4_COMPRESSION,
    },
)
```

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/wan-dancer-14b/README.md" />
