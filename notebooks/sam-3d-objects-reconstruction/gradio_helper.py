"""
Gradio demo helper for SAM-3D-Objects 3D reconstruction with OpenVINO.

Provides a `make_demo` function that creates a Gradio Blocks interface
for interactive 3D object reconstruction.
"""

import io
import warnings

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")


def _render_3d_scatter(gs_output, elev=25, azim=45):
    """Render a 3D Gaussian Splat as a matplotlib scatter plot image."""
    if gs_output is None:
        return None

    xyz = gs_output._xyz.detach().cpu().numpy()

    # Color from SH features or height
    if hasattr(gs_output, "_features_dc") and gs_output._features_dc is not None:
        colors = gs_output._features_dc.detach().cpu().numpy().squeeze()
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8)
        if colors.ndim == 2 and colors.shape[1] >= 3:
            point_colors = colors[:, :3]
        else:
            point_colors = xyz[:, 1]
    else:
        point_colors = xyz[:, 1]

    fig = plt.figure(figsize=(10, 8))

    # 3D view
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=point_colors, s=0.3, alpha=0.4)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"3D Gaussian Splat ({len(xyz):,} pts)")
    ax1.view_init(elev=elev, azim=azim)

    # Top-down projection
    ax2 = fig.add_subplot(122)
    ax2.scatter(xyz[:, 0], xyz[:, 2], s=0.2, alpha=0.3, c="steelblue")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_title("Top-down (XZ)")
    ax2.set_aspect("equal")

    plt.tight_layout()

    # Convert to numpy image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf))


def make_demo(ov_pipeline, image_folder, helper, stage1_steps=2, stage2_steps=2):
    """
    Create a Gradio Blocks demo for SAM-3D-Objects 3D reconstruction.

    Parameters
    ----------
    ov_pipeline : OVInferencePipelinePointMap
        The OpenVINO-accelerated pipeline.
    image_folder : Path
        Path to the demo image folder containing images and masks.
    helper : module
        The sam_3d_objects_helper module.
    stage1_steps : int
        Number of Stage 1 inference steps (fewer = faster).
    stage2_steps : int
        Number of Stage 2 inference steps (fewer = faster).
    """
    # Pre-load demo data
    demo_image, _ = helper.load_test_image(image_folder, index=0)
    all_masks = helper.load_test_masks(image_folder)
    n_masks = len(all_masks)

    def _create_overlay(img, mask_idx):
        """Create image + mask overlay."""
        if mask_idx < 0 or mask_idx >= n_masks:
            return img[..., :3]
        mask = all_masks[mask_idx]
        overlay = img[..., :3].copy()
        mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        mask_rgba[mask] = [255, 0, 0, 128]
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(overlay)
        ax.imshow(mask_rgba)
        ax.set_title(f"Selected mask (index={mask_idx}, coverage={mask.sum() / mask.size * 100:.1f}%)")
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return np.array(Image.open(buf))

    def on_mask_select(mask_idx):
        """Show mask overlay when mask index changes."""
        return _create_overlay(demo_image, mask_idx)

    def run_reconstruction(mask_idx, s1_steps, s2_steps, progress=gr.Progress()):
        """Run the full 3D reconstruction pipeline."""
        if mask_idx < 0 or mask_idx >= n_masks:
            return None, "Please select a valid mask index."

        mask = all_masks[mask_idx]
        mask_uint8 = mask.astype(np.uint8) * 255
        rgba = np.concatenate([demo_image[..., :3], mask_uint8[..., None]], axis=-1)

        progress(0.1, desc="Running Stage 1 (Sparse Structure) …")
        import time

        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = ov_pipeline.run(
                rgba,
                None,
                seed=42,
                stage1_only=False,
                with_mesh_postprocess=False,
                with_texture_baking=False,
                with_layout_postprocess=False,
                stage1_inference_steps=int(s1_steps),
                stage2_inference_steps=int(s2_steps),
            )

        elapsed = time.time() - t0
        progress(0.9, desc="Rendering 3D visualization …")

        if "gs" in output and output["gs"] is not None:
            gs = output["gs"]
            n_points = gs._xyz.shape[0]
            result_img = _render_3d_scatter(gs)
            status = f"Reconstruction complete: {n_points:,} Gaussian points in {elapsed:.1f}s"
        elif "coords" in output and output["coords"] is not None:
            coords = output["coords"]
            status = f"Stage 1 complete: {coords.shape[0]} voxels in {elapsed:.1f}s (Stage 2 may have failed)"
            result_img = None
        else:
            status = f"Pipeline completed in {elapsed:.1f}s but produced no output."
            result_img = None

        return result_img, status

    # Build Gradio UI
    with gr.Blocks(title="SAM-3D-Objects — OpenVINO 3D Reconstruction") as demo:
        gr.Markdown(
            "## SAM-3D-Objects — 3D Object Reconstruction with OpenVINO\n\n"
            "Select a mask index from the demo scene, adjust inference steps, "
            "then click **Reconstruct 3D** to run the full two-stage pipeline.\n\n"
            "> **Note**: Reconstruction takes ~60-120 seconds depending on hardware."
        )

        with gr.Row():
            with gr.Column(scale=1):
                mask_slider = gr.Slider(
                    minimum=0,
                    maximum=n_masks - 1,
                    step=1,
                    value=14,
                    label=f"Mask Index (0–{n_masks - 1})",
                )
                s1_slider = gr.Slider(minimum=1, maximum=12, step=1, value=stage1_steps, label="Stage 1 Steps")
                s2_slider = gr.Slider(minimum=1, maximum=12, step=1, value=stage2_steps, label="Stage 2 Steps")
                reconstruct_btn = gr.Button("Reconstruct 3D", variant="primary")
                status_text = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                with gr.Tab("Input + Mask"):
                    input_preview = gr.Image(label="Image + Mask Overlay", type="numpy", interactive=False)
                with gr.Tab("3D Reconstruction"):
                    output_img = gr.Image(label="3D Gaussian Splat", type="numpy", interactive=False)

        # Event handlers
        mask_slider.change(on_mask_select, inputs=[mask_slider], outputs=[input_preview])
        reconstruct_btn.click(
            run_reconstruction,
            inputs=[mask_slider, s1_slider, s2_slider],
            outputs=[output_img, status_text],
        )

        # Initialize with default mask
        demo.load(lambda: _create_overlay(demo_image, 14), outputs=[input_preview])

    return demo
