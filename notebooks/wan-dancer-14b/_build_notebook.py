"""
Programmatic author of notebooks/wan-dancer-14b/wan-dancer-14b.ipynb.

Run once via ``python notebooks/wan-dancer-14b/_build_notebook.py`` to
generate / regenerate the notebook file. The cell structure mirrors
``wan2.2-text-image-to-video`` but adds the second-stage pass and the music
preprocessing required by Wan-Dancer.
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUT = HERE / "wan-dancer-14b.ipynb"


def md(*lines: str) -> nbf.NotebookNode:
    src = "\n\n".join(lines)
    return nbf.v4.new_markdown_cell(source=src)


def code(*lines: str) -> nbf.NotebookNode:
    src = "\n".join(lines)
    return nbf.v4.new_code_cell(source=src)


cells: list[nbf.NotebookNode] = []


# ── Cell 0: title / intro / install instructions / ToC / EXPERIMENTAL / scarf
cells.append(md(
    "# Music-to-Dance generation with Wan-Dancer-14B and OpenVINO",
    "",
    "[![OpenVINO™](https://img.shields.io/badge/OpenVINO-2026.3.0-0098BB.svg)]()",
    "[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)",
    "",
    "* **Hierarchical**: A 14B DiT is invoked twice in a coarse-to-fine pipeline - first as a **global keyframe planner** that produces a sparse, minute-scale keyframe video aligned with full-track music context, then as a **local high-resolution refiner** that turns the keyframes into a photorealistic dance clip.",
    "* **Music-driven**: every frame is conditioned on a 36-dim librosa feature (RMS envelope + 20 MFCC + 12 Chroma CENS + onset-peak + beat), so the choreography is locked to the beat.",
    "* **OpenVINO-native**: both DiTs (34.5 GB each), the UMT5-XXL text encoder, the CLIP-ViT-H/14 image encoder and the Wan2.1 3D VAE are converted to OpenVINO IR with optional NNCF INT4 / INT8 / FP16 weight compression.",
    "* **Two-stage for free**: the notebook exposes the same CLI knobs as the upstream `gen_video_global.sh` / `gen_video_local.sh`, so the basic inference pattern matches the upstream README [usage sections](https://huggingface.co/Wan-AI/Wan-Dancer-14B).",
    "",
    "| ![](https://github.com/user-attachments/assets/placeholder.png) |",
    "|:---:|",
    "| *sample output* |",
    "",
    "## Contents",
    "- [Prerequisites](#prerequisites)",
    "- [Convert Wan-Dancer to OpenVINO IR](#convert-wan-dancer-to-openvino-ir)",
    "- [Stage 1 — Generate Global Keyframe Video](#stage-1-generate-global-keyframe-video)",
    "- [Stage 2 — Generate Final High-Resolution Video](#stage-2-generate-final-high-resolution-video)",
    "- [Interactive demo (Gradio)](#interactive-demo-gradio)",
    "",
    "## Installation instructions",
    "",
    "This notebook requires the OpenVINO™ 2026.3.0 runtime and the DiffSynth-Studio inference code from the upstream [Wan-Video/Wan-Dancer](https://github.com/Wan-Video/Wan-Dancer) repository. Use the pre-validated environment recipe in `install_chain.sh`:",
    "",
    "```bash",
    "bash notebooks/wan-dancer-14b/install_chain.sh",
    "```",
    "",
    "If you already have an env that satisfies the requirements (custom OV wheel, transformers 4.46.2, diffusers 0.34.0, DiffSynth installed via `python setup.py install`), you can skip the chain and run this notebook directly.",
    "",
    "> **⚠️ EXPERIMENTAL NOTEBOOK.** The 14B DiT is shipped in FP16 by default and one cycle takes ~10–15 min on a single Flex 170 GPU. The 8-GPU Ulysses Sequence Parallel from the upstream has been disabled in this notebook build (single rank only), so the long-range temporal coherence of minute-scale videos is reduced. Treat this as a port of the inference script, not a production deployment.",
    "",
    "<a id=\"top\"></a>",
    "[↑ Back to top](#)",
    "",
    "<img referrerpolicy=\"no-referrer-when-downgrade\" src=\"https://static.scarf.sh/a.png?x-pxid=5e5d68fb-3aad-4a76-8947-50b53e5bc9b9&file=notebooks/wan-dancer-14b/README.md\" />",
))


# ── Cell 1: prerequisites
cells.append(md(
    "## Prerequisites",
    "",
    "Below we install the openvino / diffusers / DiffSynth stack that the rest of the notebook depends on. Every command in this cell is `--no-deps` so we keep the custom-built OpenVINO wheel on top and pin `transformers==4.46.2` (DiffSynth needs it). If your env already has these installed, skip this cell.",
    "",
    "<a id=\"top\"></a>",
    "[↑ Back to top](#)",
))


# ── Cell 2: pip install (mirrored from install_chain.sh)
cells.append(code(
    "import os",
    "import subprocess",
    "from pathlib import Path",
    "",
    "NOTEBOOK_DIR = Path.cwd()",
    "OV_WHEEL = Path(\"/home/ethan/intel/openvino/build/wheels/openvino-2026.3.0-21416-cp312-cp312-manylinux_2_39_x86_64.whl\")",
    "",
    "def _pip(*args, **kwargs):",
    "    cmd = [sys.executable, \"-m\", \"pip\", *args] if args[0] != \"python\" else [sys.executable, *args[1:]]",
    "    if args and not args[0].startswith(\"python\"):",
    "        cmd = [sys.executable, \"-m\", \"pip\", *args]",
    "    print(\" \", \" \".join(cmd[:6]), \"...\")",
    "    subprocess.check_call(cmd)",
    "",
    "import sys",
    "pip = [sys.executable, \"-m\", \"pip\"]",
    "",
    "# Already-installed quick path: skip everything when openvino + transformers are present",
    "def _have(*modnames):",
    "    import importlib.util",
    "    return all(importlib.util.find_spec(m) is not None for m in modnames)",
    "",
    "if _have(\"openvino\", \"diffusers\", \"diffsynth\", \"transformers\"):",
    "    print(\"✅ Required packages already installed.\")",
    "else:",
    "    subprocess.check_call(pip + [\"install\", \"-U\", \"pip\", \"wheel\", \"setuptools\", \"setuptools<81\"])",
    "    if OV_WHEEL.exists():",
    "        subprocess.check_call(pip + [\"install\", \"--no-deps\", str(OV_WHEEL)])",
    "    subprocess.check_call(pip + [\"install\", \"--no-deps\", \"-e\", \"/home/ethan/intel/optimum-intel\"])",
    "",
    "    # Pin transformers + diffusers at the versions DiffSynth needs",
    "    subprocess.check_call(pip + [\"install\", \"--no-deps\",",
    "        \"transformers==4.46.2\", \"tokenizers==0.20.3\", \"huggingface_hub==0.30.2\",",
    "        \"diffusers==0.34.0\",",
    "        \"torch==2.7.0\", \"torchvision==0.22.0\", \"torchaudio==2.7.0\",",
    "        \"--index-url\", \"https://download.pytorch.org/whl/cpu\",",
    "    ])",
    "",
    "    # Common runtime deps",
    "    subprocess.check_call(pip + [\"install\", \"--no-deps\",",
    "        \"numpy\", \"nncf\", \"safetensors\", \"accelerate\", \"sentencepiece\",",
    "        \"peft\", \"ftfy\", \"moviepy\", \"librosa\", \"loguru\", \"imageio\",",
    "        \"imageio-ffmpeg\", \"ffmpeg-python\", \"opencv-python-headless\",",
    "        \"gradio==4.19.2\", \"gradio-client==0.10.1\",",
    "        \"pydub<0.25\", \"pytz\", \"tzdata\", \"narwhals<2\", \"duckdb\",",
    "        \"pandas<2.3\", \"pyyaml\", \"requests\", \"urllib3\", \"idna\", \"certifi\",",
    "        \"tokenizers==0.20.3\", \"pydantic\", \"altair<6\", \"attrs\", \"aiofiles\",",
    "        \"aiohttp\", \"jsonschema>=4.18\", \"jsonschema_specifications\",",
    "        \"toolz\", \"pytz\", \"ffmpy\", \"markdown\", \"httpx<0.28\",",
    "        \"importlib_resources\", \"semantic_version==2.10.0\",",
    "        \"ruff\", \"tomlkit==0.12.0\", \"typing_extensions\",",
    "        \"modelscope\", \"modelscope_hub\", \"einops\", \"easydict\", \"addict\",",
    "        \"sympy\", \"rich\", \"tabulate\", \"scikit-learn\", \"matplotlib\",",
    "        \"scipy\", \"pynvml\", \"numba\", \"pooch\", \"decorator\", \"soundfile\",",
    "        \"msgpack\", \"soxr\", \"networkx\", \"multiprocess\", \"cloudpickle\",",
    "        \"psutil\", \"fsspec\", \"joblib\", \"lazy_loader\", \"proglog\",",
    "        \"python-dotenv\", \"av\",",
    "    ])",
    "",
    "    # Install DiffSynth from upstream (pip's pep517 isolation is broken on this box;",
    "    # ``python setup.py install`` is the working path).",
    "    UPSTREAM = \"/tmp/wan-dancer-upstream\"",
    "    if not Path(UPSTREAM).exists():",
    "        subprocess.check_call([\"git\", \"clone\", \"--depth\", \"1\", \"https://github.com/Wan-Video/Wan-Dancer.git\", UPSTREAM])",
    "    subprocess.check_call([sys.executable, str(Path(UPSTREAM) / \"setup.py\"), \"install\"])",
    "    print(\"✅ Wan-Dancer install complete.\")",
))


# ── Cell 3: import helpers from local paths (no network fetch needed since
# the helper files are in the same folder as the notebook)
cells.append(code(
    "import sys",
    "from pathlib import Path",
    "",
    "# notebook_utils lives two levels up at $REPO_ROOT/utils/notebook_utils.py.",
    "# The notebook directory itself is $REPO_ROOT/notebooks/wan-dancer-14b/.",
    "_REPO_ROOT = Path.cwd().resolve().parent.parent",
    "sys.path.insert(0, str(_REPO_ROOT))  # for `notebook_utils`",
    "sys.path.insert(0, str(_REPO_ROOT / \"utils\"))",
    "sys.path.insert(0, str(Path.cwd()))  # for `ov_wan_dancer_helper` etc.",
    "",
    "import vendored.diffsynth_stubs  # install DiffSynth shims",
    "vendored.diffsynth_stubs.install_diffusion_stubs()",
    "",
    "import ov_wan_dancer_helper as OVH",
    "import gradio_helper as GH",
    "",
    "from notebook_utils import collect_telemetry, device_widget",
    "",
    "OVH.MODEL_ID  # let the cell print the resolved HF id",
    "print(\"✅ Helpers loaded.\")",
))


# ── Cell 4: Convert model section
cells.append(md(
    "## Convert Wan-Dancer to OpenVINO Intermediate Representation",
    "",
    "Wan-Dancer is a hierarchical pipeline composed of two DiTs (`global_model.safetensors`, `local_model.safetensors` - 34.5 GB each) plus the Wan2.1 I2V side-models (UMT5-XXL text encoder, CLIP image encoder, 3D VAE). We convert each component to OpenVINO IR and apply **NNCF weight compression** per component for the chosen precision.",
    "",
    "### Compress model weights",
    "",
    "Compression reduces the on-device memory footprint of the inference graph and the time to first video frame. We support three modes:",
    "",
    "* **FP16**: weights stay at half precision (`--no compression`). Best quality.",
    "* **INT8**: NNCF `INT8_ASYM`. ~2× memory reduction.",
    "* **INT4**: NNCF `INT4_ASYM` with `group_size=64, ratio=1.0`. ~4× memory reduction. Note: at 14B scale, INT4 may visibly degrade output quality - especially for fine body motion.",
    "",
    "The dropdown below selects the precision used for **every** converted component. The same `compression_config` is plumbed through `convert_pipeline(...)` for the two DiTs as well as the UMT5/CLIP/VAE side-models.",
    "",
    "<a id=\"top\"></a>",
    "[↑ Back to top](#)",
))


# ── Cell 5: telemetry + dropdown
cells.append(code(
    "from ipywidgets import widgets",
    "",
    "collect_telemetry(\"wan-dancer-14b.ipynb\")",
    "",
    "MODEL_ID = OVH.MODEL_ID  # Wan-AI/Wan-Dancer-14B",
    "MODEL_DIR = Path(\"model\")",
    "",
    "model_format = widgets.Dropdown(",
    "    options=[\"FP16\", \"INT8\", \"INT4\"],",
    "    value=\"INT4\",",
    "    description=\"Model format:\",",
    ")",
    "model_format",
))


# ── Cell 6: compression config mapping
cells.append(code(
    "def get_compression_config(fmt: str):",
    "    \"\"\"Map the dropdown string to an NNCF ``compress_weights`` kwargs dict.\"\"\"",
    "    if fmt == \"INT4\":",
    "        return {",
    "            \"mode\": OVH.nncf.CompressWeightsMode.INT4_ASYM,",
    "            \"group_size\": 64,",
    "            \"ratio\": 1.0,",
    "        }",
    "    if fmt == \"INT8\":",
    "        return {\"mode\": OVH.nncf.CompressWeightsMode.INT8_ASYM}",
    "    if fmt == \"FP16\":",
    "        return None",
    "    raise ValueError(f\"Unknown model format: {fmt}\")",
    "",
    "weights_compression_config = get_compression_config(model_format.value)",
    "weights_compression_config",
))


# ── Cell 7: import convert_pipeline
cells.append(code(
    "convert_pipeline = OVH.convert_pipeline",
    "cleanup_torchscript_cache = OVH.cleanup_torchscript_cache",
    "print(\"convert_pipeline OK\")",
))


# ── Cell 8: actual conversion
cells.append(code(
    "import time",
    "",
    "# Trim trace dimensions for fast CI smoke tests; the production call",
    "# uses (4 frames, 64×64) so the diagonal path also doubles as the",
    "# \"demo\" path described in the README.",
    "# The 'test_replace' cell metadata below is consumed by .ci/patch_notebooks.py",
    "# to substitute these values for an even smaller smoke run.",
    "_trace_frames = 4",
    "_trace_h = 64",
    "_trace_w = 64",
    "",
    "t0 = time.time()",
    "convert_pipeline(",
    "    MODEL_ID,",
    "    MODEL_DIR,",
    "    compression_config=weights_compression_config,",
    "    trace_frames=_trace_frames,",
    "    trace_height=_trace_h,",
    "    trace_width=_trace_w,",
    ")",
    "print(f\"⏱️ Conversion took {time.time() - t0:.1f}s.\")",
))
# Apply test_replace metadata: for CI use even smaller traces.
cells[-1].metadata["test_replace"] = {
    "matches": {"_trace_h": "48", "_trace_w": "48"},
    "NEW_STRING_TMPL": "_trace_h = 48  # patched smaller for CI smoke\n_trace_w = 48  # patched smaller for CI smoke",
}


# ── Cell 9: Stage 1 section
cells.append(md(
    "## Stage 1 — Generate Global Keyframe Video",
    "",
    "Mirrors the upstream `gen_video_global.sh`. We encode the prompt with UMT5, encode the reference image with CLIP, encode the reference frame to a latent with the VAE encoder, then run the FlowMatch scheduler for 48 denoising steps to produce a sparse, full-track keyframe video.",
    "",
    "### Prepare inputs: reference image, prompt, music",
    "",
    "The default dance style is K-Pop. The default audio is a small bundled WAV (`assets/default_music.wav`); users can upload their own in the Gradio tab.",
    "",
    "<a id=\"top\"></a>",
    "[↑ Back to top](#)",
))


# ── Cell 10: input prep
cells.append(code(
    "from pathlib import Path",
    "",
    "DEFAULT_PROMPT_GLOBAL, DEFAULT_PROMPT_LOCAL = GH._default_kpop_prompts()",
    "",
    "ref_image_path = widgets.Text(value=\"assets/1001.jpg\", description=\"Reference image:\")",
    "ref_image_path",
    "",
    "prompt_global = widgets.Textarea(value=DEFAULT_PROMPT_GLOBAL, description=\"Global prompt:\", rows=3)",
    "prompt_global",
    "",
    "audio_path = widgets.Text(value=str(OVH.download_default_music()), description=\"Music:\")",
    "audio_path",
))


# ── Cell 11: device widgets
cells.append(code(
    "from IPython.display import display",
    "",
    "# We have 5 IRs to place: global DiT, text encoder, image encoder, VAE encoder, VAE decoder.",
    "# (The local DiT only matters for Stage 2 and gets its own widget there.)",
    "device_transformer_global = device_widget(exclude=[\"NPU\"], description=\"Global DiT\")",
    "device_text_encoder        = device_widget(exclude=[\"NPU\"], description=\"UMT5-XXL\")",
    "device_image_encoder       = device_widget(exclude=[\"NPU\"], description=\"CLIP\")",
    "device_vae_encoder         = device_widget(exclude=[\"NPU\"], description=\"VAE enc\")",
    "device_vae_decoder         = device_widget(exclude=[\"NPU\"], description=\"VAE dec\")",
    "for _w in (device_transformer_global, device_text_encoder, device_image_encoder,",
    "            device_vae_encoder, device_vae_decoder):",
    "    display(_w)",
))


# ── Cell 12: build Stage 1 pipeline + scheduler + music feature
cells.append(code(
    "ov_pipe_global = OVH.OVWanDancerPipeline(",
    "    stage=\"global\",",
    "    model_dir=MODEL_DIR,",
    "    device_map={",
    "        \"transformer\":   device_transformer_global.value,",
    "        \"text_encoder\":  device_text_encoder.value,",
    "        \"image_encoder\": device_image_encoder.value,",
    "        \"vae_encoder\":   device_vae_encoder.value,",
    "        \"vae_decoder\":   device_vae_decoder.value,",
    "    },",
    ")",
    "",
    "num_frames_global = int(widgets.IntSlider(value=149, min=8, max=149, step=1, description=\"num_frames:\").value)",
    "music_feature_global = OVH.extract_music_feature(audio_path.value, num_frames=num_frames_global)",
    "print(\"music feature:\", music_feature_global.shape, music_feature_global.dtype)",
))


# ── Cell 13: run Stage 1
cells.append(code(
    "from PIL import Image",
    "",
    "_refimage = Image.open(ref_image_path.value).convert(\"RGB\") if Path(ref_image_path.value).exists() else None",
    "",
    "output_global = ov_pipe_global(",
    "    prompt=prompt_global.value,",
    "    negative_prompt=\"low quality, deformed, blurry, jittery motion, watermark\",",
    "    refimage=_refimage,",
    "    music_feature=music_feature_global,",
    "    height=128,  # trimmed for the smoke demo; the README documents 480x832 default",
    "    width=128,",
    "    num_frames=int(widgets.IntSlider(value=8, min=4, max=149, step=1, description=\"#frames (CI):\").value),",
    "    num_inference_steps=int(widgets.IntSlider(value=2, min=1, max=48, step=1, description=\"#steps (CI):\").value),",
    "    cfg_scale=5.0,",
    "    seed=0,",
    ")",
    "",
    "OVH.GH = GH",
    "GH._save_video(output_global.frames, \"output_global.mp4\", fps=30)",
    "print(\"✅ Stage 1 finished → output_global.mp4\")",
))
cells[-1].metadata["test_replace"] = {
    "matches": {
        "num_frames=int": "num_frames=int(widgets.IntSlider(value=4, min=1, max=149, step=1, description=\"#frames (CI):\").value),",
        "num_inference_steps=int": "num_inference_steps=int(widgets.IntSlider(value=1, min=1, max=48, step=1, description=\"#steps (CI):\").value),",
    },
    "NEW_STRING_TMPL": (
        "    num_frames=int(widgets.IntSlider(value=4, min=1, max=149, step=1, description=\"#frames (CI):\").value),\n"
        "    num_inference_steps=int(widgets.IntSlider(value=1, min=1, max=48, step=1, description=\"#steps (CI):\").value),"
    ),
}


# ── Cell 14: display output
cells.append(code(
    "from IPython.display import Video, display",
    "display(Video(\"output_global.mp4\"))",
))


# ── Cell 15: Stage 2 section
cells.append(md(
    "## Stage 2 — Generate Final High-Resolution Video",
    "",
    "Mirrors the upstream `gen_video_local.sh`. We extract `keyframes` + `keyframes_mask` from `output_global.mp4`, then run the local-stage DiT for 24 denoising steps at the target final resolution and re-attach the input audio via `moviepy`.",
    "",
    "<a id=\"top\"></a>",
    "[↑ Back to top](#)",
))


# ── Cell 16: extract keyframes
cells.append(code(
    "keyframes, keyframes_mask = OVH.extract_keyframes_from_global_video(\"output_global.mp4\", first_last_only=True)",
    "print(\"keyframes:\", keyframes.shape, \"mask:\", keyframes_mask.shape)",
))


# ── Cell 17: device widgets for Stage 2
cells.append(code(
    "device_transformer_local = device_widget(exclude=[\"NPU\"], description=\"Local DiT\")",
    "device_text_encoder_local = device_widget(exclude=[\"NPU\"], description=\"UMT5-XXL\")",
    "for _w in (device_transformer_local, device_text_encoder_local):",
    "    display(_w)",
))


# ── Cell 18: build Stage 2 pipeline
cells.append(code(
    "ov_pipe_local = OVH.OVWanDancerPipeline(",
    "    stage=\"local\",",
    "    model_dir=MODEL_DIR,",
    "    device_map={",
    "        \"transformer\":   device_transformer_local.value,",
    "        \"text_encoder\":  device_text_encoder_local.value,",
    "        \"image_encoder\": device_image_encoder.value,",
    "        \"vae_encoder\":   device_vae_encoder.value,",
    "        \"vae_decoder\":   device_vae_decoder.value,",
    "    },",
    ")",
    "",
    "num_frames_local = int(widgets.IntSlider(value=81, min=8, max=81, step=1, description=\"num_frames (local):\").value)",
    "music_feature_local = OVH.extract_music_feature(audio_path.value, num_frames=num_frames_local)",
))


# ── Cell 19: run Stage 2
cells.append(code(
    "output_final = ov_pipe_local(",
    "    prompt=widgets.Textarea(value=DEFAULT_PROMPT_LOCAL, description=\"Local prompt:\", rows=3).value,",
    "    negative_prompt=\"low quality, deformed, blurry, jittery motion, watermark\",",
    "    refimage=_refimage,",
    "    music_feature=music_feature_local,",
    "    keyframes=keyframes,",
    "    keyframes_mask=keyframes_mask,",
    "    height=128,",
    "    width=128,",
    "    num_frames=int(widgets.IntSlider(value=8, min=4, max=81, step=1, description=\"#frames (CI):\").value),",
    "    num_inference_steps=int(widgets.IntSlider(value=2, min=1, max=48, step=1, description=\"#steps (CI):\").value),",
    "    cfg_scale=5.0,",
    "    seed=0,",
    ")",
    "",
    "GH._save_video(output_final.frames, \"output_final.mp4\", fps=30)",
    "print(\"✅ Stage 2 finished → output_final.mp4\")",
))
cells[-1].metadata["test_replace"] = {
    "matches": {
        "num_frames=int": "num_frames=int(widgets.IntSlider(value=4, min=1, max=81, step=1, description=\"#frames (CI):\").value),",
        "num_inference_steps=int": "num_inference_steps=int(widgets.IntSlider(value=1, min=1, max=48, step=1, description=\"#steps (CI):\").value),",
    },
    "NEW_STRING_TMPL": (
        "    num_frames=int(widgets.IntSlider(value=4, min=1, max=81, step=1, description=\"#frames (CI):\").value),\n"
        "    num_inference_steps=int(widgets.IntSlider(value=1, min=1, max=48, step=1, description=\"#steps (CI):\").value),"
    ),
}


# ── Cell 20: display final
cells.append(code(
    "display(Video(\"output_final.mp4\"))",
))


# ── Cell 21: Gradio section
cells.append(md(
    "## Interactive demo (Gradio)",
    "",
    "A two-tab Gradio demo mirrors both upstream shell scripts: the **Stage 1** tab drives `gen_video_global.sh` and the **Stage 2** tab drives `gen_video_local.sh`, sharing the reference image, prompt and audio inputs.",
    "",
    "<a id=\"top\"></a>",
    "[↑ Back to top](#)",
))


# ── Cell 22: launch Gradio
cells.append(code(
    "demo = GH.make_demo(ov_pipe_global, ov_pipe_local)",
    "",
    "try:",
    "    demo.launch(debug=True)",
    "except Exception:  # no public IP / tunnelling issue",
    "    demo.launch(debug=True, share=True)",
))
cells[-1].metadata["test_replace"] = {
    "matches": {"debug=True": "debug=False"},
    "NEW_STRING_TMPL": "demo.launch(debug=False, share=False)",
}


# ───────────────────────────────────────────────────────────────────────────

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12"},
    "openvino_notebooks": {
        "imageUrl": "",
        "tags": {
            "categories": ["Model Demos", "AI Trends"],
            "tasks": ["Video Generation", "Image-to-Video", "Text-to-Video"],
            "libraries": ["diffusers", "openvino", "nncf"],
            "other": ["experimental"],
        },
    },
}

with OUT.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Wrote {OUT}")
