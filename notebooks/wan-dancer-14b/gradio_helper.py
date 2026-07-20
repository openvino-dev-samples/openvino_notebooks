# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
gradio_helper for the Wan-Dancer-14B OpenVINO notebook.

Two-tab Gradio UI matching the upstream pipeline:

* **Stage 1 — Global Keyframe Video** mirrors ``gen_video_global.sh``.
* **Stage 2 — Final High-Resolution Video** mirrors ``gen_video_local.sh``
  and consumes the output of Stage 1 as ``keyframes`` conditioning.

A shared input row at the top accepts the reference image, the dance-style
prompt and (optionally) an uploaded music file. The default values are
loaded from ``assets/default_music.wav`` plus the upstream K-Pop dance
prompt shipped at ``assets/kpop_global.txt`` if present, otherwise a
sensible built-in string.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

# -- Sample preset strings (mirror the upstream ``gen_video/prompt/*.txt``) --

DEFAULT_PROMPT_GLOBAL = "K-Pop dance cover, dynamic camera, vibrant stage lighting, " "synchronised choreography, full-body motion, high energy."

DEFAULT_PROMPT_LOCAL = "K-Pop dance cover, photorealistic, smooth motion, " "studio-quality motion blur, dynamic camera, vibrant stage lighting."

DEFAULT_REF_IMAGE_URL = "https://huggingface.co/datasets/Wan-AI/Wan-Dancer-14B-sample/resolve/main/" "ref_image_3001.jpg"


def _maybe_read_prompt(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _default_kpop_prompts() -> Tuple[str, str]:
    here = Path(__file__).resolve().parent
    global_prompt = _maybe_read_prompt(here / "assets" / "kpop_global.txt") or DEFAULT_PROMPT_GLOBAL
    local_prompt = _maybe_read_prompt(here / "assets" / "kpop_local.txt") or DEFAULT_PROMPT_LOCAL
    return global_prompt, local_prompt


def _default_audio_path() -> Optional[str]:
    path = Path(__file__).resolve().parent / "assets" / "default_music.wav"
    return str(path) if path.exists() else None


# -- Helpers ----------------------------------------------------------------


def _resize_to_canvas(pil_image: Image.Image, height: int, width: int) -> Image.Image:
    """Resize a PIL image so its longer side matches ``height``/``width`` while
    preserving aspect ratio, then centre-pad to exactly ``(height, width)``.
    """
    if pil_image is None:
        pil_image = Image.new("RGB", (width, height), color=(127, 127, 127))
    pil_image = pil_image.convert("RGB")
    src_w, src_h = pil_image.size
    ratio = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * ratio)))
    new_h = max(1, int(round(src_h * ratio)))
    pil_image = pil_image.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (width, height), color=(127, 127, 127))
    canvas.paste(pil_image, ((width - new_w) // 2, (height - new_h) // 2))
    return canvas


# -- Demo assembly ----------------------------------------------------------


def make_demo(global_pipeline, local_pipeline):
    """Construct a Gradio demo with both stages wired up.

    Parameters
    ----------
    global_pipeline : ``OVWanDancerPipeline``
        Already-built Stage 1 (global) pipeline. We import lazily here so the
        import order in the notebook never blocks Gradio from loading.
    local_pipeline : ``OVWanDancerPipeline``
        Already-built Stage 2 (local) pipeline.
    """
    from ov_wan_dancer_helper import extract_music_feature  # local import
    from moviepy.editor import AudioFileClip, ImageSequenceClip

    default_global_prompt, default_local_prompt = _default_kpop_prompts()

    with gr.Blocks(title="Wan-Dancer-14B OpenVINO", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Music-to-Dance Generation with Wan-Dancer-14B + OpenVINO\n\n"
            "Hierarchical 2-stage generation: Stage 1 produces a sparse "
            "global keyframe video from music + reference image + prompt; "
            "Stage 2 refines it into the final high-resolution dance clip."
        )

        # ----- shared inputs -----
        with gr.Row():
            ref_image_input = gr.Image(value=DEFAULT_REF_IMAGE_URL, type="pil", label="Reference image")
            audio_input = gr.Audio(
                value=_default_audio_path(),
                type="filepath",
                label="Music (default bundled)",
            )

        prompt_global_state = gr.State(default_global_prompt)
        prompt_local_state = gr.State(default_local_prompt)

        with gr.Tab("Stage 1 — Global Keyframe Video"):
            gr.Markdown("Mirrors ``gen_video_global.sh``: 48 diffusion steps, CFG 5.0, " "30 fps; output drives the keyframes for Stage 2.")
            with gr.Row():
                prompt_global = gr.Textbox(
                    label="Global prompt (dance style)",
                    lines=2,
                    value=default_global_prompt,
                )
                neg_prompt_global = gr.Textbox(
                    label="Negative prompt",
                    lines=2,
                    value="low quality, deformed, blurry, jittery motion",
                )
            with gr.Row():
                num_frames_global = gr.Slider(8, 149, step=1, value=149, label="num_frames (30 fps)")
                steps_global = gr.Slider(1, 48, step=1, value=48, label="num_inference_steps")
                cfg_global = gr.Slider(1.0, 10.0, step=0.1, value=5.0, label="guidance_scale")
            with gr.Row():
                height = gr.Slider(64, 1024, step=16, value=480, label="height")
                width = gr.Slider(64, 1024, step=16, value=832, label="width")
                seed_global = gr.Slider(0, 2**32 - 1, step=1, value=0, label="seed")
            btn_global = gr.Button("Run Stage 1", variant="primary")
            out_global = gr.Video(label="output_global.mp4", show_label=True)
            download_global = gr.File(label="Download output_global.mp4")
            btn_global.click(
                fn=lambda *a: _run_global(global_pipeline, *a),
                inputs=[
                    ref_image_input,
                    audio_input,
                    prompt_global,
                    neg_prompt_global,
                    num_frames_global,
                    steps_global,
                    cfg_global,
                    height,
                    width,
                    seed_global,
                ],
                outputs=[out_global, download_global],
            )

        with gr.Tab("Stage 2 — Final High-Resolution Video"):
            gr.Markdown(
                "Mirrors ``gen_video_local.sh``: takes the Stage 1 "
                "``output_global.mp4`` as keyframe conditioning, runs 24 diffusion "
                "steps at the desired final resolution with audio re-attachment."
            )
            with gr.Row():
                prompt_local = gr.Textbox(
                    label="Local prompt (dance style)",
                    lines=2,
                    value=default_local_prompt,
                )
                neg_prompt_local = gr.Textbox(
                    label="Negative prompt",
                    lines=2,
                    value="low quality, deformed, blurry, jittery motion",
                )
            with gr.Row():
                global_video = gr.Video(
                    label="Stage 1 output (input to Stage 2)",
                    sources=["upload", "clipboard"],
                )
            with gr.Row():
                num_frames_local = gr.Slider(8, 81, step=1, value=81, label="num_frames (30 fps)")
                steps_local = gr.Slider(1, 48, step=1, value=24, label="num_inference_steps")
                cfg_local = gr.Slider(1.0, 10.0, step=0.1, value=5.0, label="guidance_scale")
            with gr.Row():
                height_local = gr.Slider(64, 1024, step=16, value=480, label="height")
                width_local = gr.Slider(64, 1024, step=16, value=832, label="width")
                seed_local = gr.Slider(0, 2**32 - 1, step=1, value=0, label="seed")
            btn_local = gr.Button("Run Stage 2", variant="primary")
            out_final = gr.Video(label="output_final.mp4", show_label=True)
            download_final = gr.File(label="Download output_final.mp4")
            btn_local.click(
                fn=lambda *a: _run_local(local_pipeline, *a),
                inputs=[
                    ref_image_input,
                    audio_input,
                    global_video,
                    prompt_local,
                    neg_prompt_local,
                    num_frames_local,
                    steps_local,
                    cfg_local,
                    height_local,
                    width_local,
                    seed_local,
                ],
                outputs=[out_final, download_final],
            )

        gr.Markdown(
            "---\n"
            "**Tips.** Stage 1 is the slow step (large seq parallelism in "
            "the upstream; single-rank OV is slower). Stage 2 reuses "
            "Stage 1's output as keyframes. Audio is re-attached at the end "
            "via ``moviepy.editor.AudioFileClip``.\n\n"
            "**Hardware note.** The 14B DiT is shipped FP16 by default; "
            "INT4 weights trade quality for ~4× VRAM savings. The notebook "
            "dropdown lets you pick one of the three before ``convert_pipeline``."
        )

    return demo


# -- Runner functions --------------------------------------------------------


def _run_global(pipeline, refimage, audio_path, prompt, neg_prompt, num_frames, num_steps, cfg_scale, height, width, seed):
    """Stage 1 runner that returns the produced MP4 path + downloadable copy."""
    height, width = int(height), int(width)
    num_frames, num_steps = int(num_frames), int(num_steps)
    cfg_scale = float(cfg_scale)
    seed = int(seed)

    if refimage is None:
        refimage = Image.new("RGB", (width, height), color=(127, 127, 127))
    else:
        refimage = _resize_to_canvas(refimage, height, width)

    music_feature = extract_music_feature(audio_path, num_frames=num_frames) if audio_path and Path(audio_path).exists() else None

    result = pipeline(
        prompt=prompt,
        negative_prompt=neg_prompt,
        refimage=refimage,
        music_feature=music_feature,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        cfg_scale=cfg_scale,
        seed=seed,
    )
    out_path = "output_global.mp4"
    _save_video(result.frames, out_path, fps=30)
    return out_path, out_path


def _run_local(pipeline, refimage, audio_path, global_video_path, prompt, neg_prompt, num_frames, num_steps, cfg_scale, height, width, seed):
    """Stage 2 runner that consumes ``global_video_path`` and re-attaches audio."""
    from ov_wan_dancer_helper import extract_keyframes_from_global_video  # local import
    from moviepy.editor import AudioFileClip, concatenate_videoclips

    height, width = int(height), int(width)
    num_frames, num_steps = int(num_frames), int(num_steps)
    cfg_scale = float(cfg_scale)
    seed = int(seed)

    if refimage is None:
        refimage = Image.new("RGB", (width, height), color=(127, 127, 127))
    else:
        refimage = _resize_to_canvas(refimage, height, width)

    music_feature = extract_music_feature(audio_path, num_frames=num_frames) if audio_path and Path(audio_path).exists() else None

    keyframes, keyframes_mask = (None, None)
    if global_video_path is not None:
        try:
            keyframes, keyframes_mask = extract_keyframes_from_global_video(global_video_path)
        except Exception as exc:  # noqa: BLE001 - best-effort
            print(f"[Stage 2] Could not parse Stage 1 output ({exc}); " "using first-frame reference instead.")

    result = pipeline(
        prompt=prompt,
        negative_prompt=neg_prompt,
        refimage=refimage,
        music_feature=music_feature,
        keyframes=keyframes,
        keyframes_mask=keyframes_mask,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        cfg_scale=cfg_scale,
        seed=seed,
    )
    out_path = "output_final.mp4"
    _save_video(result.frames, out_path, fps=30)
    if audio_path and Path(audio_path).exists():
        try:
            _attach_audio(out_path, audio_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[Stage 2] Could not re-attach audio: {exc}")
    return out_path, out_path


def _save_video(frames_tensor, out_path, fps: int = 30):
    """Save a [1, F, 3, H, W] tensor to an MP4 via ``imageio[ffmpeg]``."""
    import imageio.v2 as imageio

    frames = frames_tensor[0].clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    frames = (frames * 255).round().astype(np.uint8)
    imageio.mimsave(out_path, list(frames), fps=fps, codec="libx264", quality=8)
    return out_path


def _attach_audio(video_path, audio_path):
    """Combine a silent video with an audio file using moviepy."""
    from moviepy.editor import VideoFileClip, AudioFileClip

    clip = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    audio = audio.subclip(0, clip.duration)
    final = clip.set_audio(audio)
    final_path = str(Path(video_path).with_name(Path(video_path).stem + "_with_audio.mp4"))
    final.write_videofile(final_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    # Replace the original with the audio-attached version so the user picks
    # up the right file from the download widget.
    import shutil

    shutil.move(final_path, video_path)
