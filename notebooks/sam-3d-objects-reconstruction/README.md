# SAM-3D-Objects Reconstruction with OpenVINO

This notebook demonstrates how to run [SAM-3D-Objects](https://github.com/facebookresearch/sam-3d-objects) (Meta's 3D reconstruction pipeline) using [OpenVINO™](https://github.com/openvinotoolkit/openvino) for optimized CPU inference.

SAM-3D-Objects takes a single RGB image and object masks as input, and generates 3D Gaussian Splat reconstructions for each detected object — suitable for real-time 3D rendering.

<p align="center">
  <img src="https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/main/docs/docs_assets/images/splash_figure-2.png" width="80%">
</p>

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [OpenVINO Optimizations](#openvino-optimizations)
- [File Structure](#file-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Model Details](#model-details)
- [Performance Notes](#performance-notes)

## Overview

The SAM-3D-Objects pipeline performs two-stage 3D reconstruction:

1. **Sparse Structure (SS) Stage**: Generates a coarse 3D occupancy grid from DINOv2-encoded image/mask features via a 24-block flow matching transformer.
2. **Structured Latent (SLat) Stage**: Decodes the sparse structure into dense per-voxel latent features, which are then decoded into 3D Gaussian Splat parameters.

This notebook converts all neural network sub-models to OpenVINO Intermediate Representation (IR), enabling efficient inference on Intel CPUs and GPUs without requiring CUDA.

## Architecture

```
Image + Mask
     │
     ▼
┌─────────────────────────────────────────┐
│  DINOv2 ViT-L/14 Backbone (merged)     │ ← Single OV model, dual outputs
│  Output 0: postnorm (for SS stage)      │
│  Output 1: prenorm  (for SLat stage)    │
└─────────────┬───────────────────────────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
┌─────────┐     ┌──────────┐
│ SS Stage│     │SLat Stage│
│ (shape) │────▶│ (detail) │
└────┬────┘     └────┬─────┘
     │               │
     ▼               ▼
  Occupancy      Gaussian Splat
  Grid           Parameters (xyz, rgb, scale, rotation, opacity)
```

## OpenVINO Optimizations

### DINOv2 Backbone Merging

All four DINOv2 embedders (SS-image, SS-mask, SLat-image, SLat-mask) share **identical** ViT-L/14 backbone weights (~1.2 GB each). We merge them into a **single** OpenVINO model with two outputs:

- **Output 0** (postnorm): Used by the Sparse Structure (SS) stage
- **Output 1** (prenorm): Used by the Structured Latent (SLat) stage

This saves **~3.6 GB** of disk space (4 copies → 1 copy).

For mask inputs, the 1-channel → 3-channel repeat is handled in Python before calling the model.

### NNCF Quantization

The notebook includes an optional section for INT8 quantization using [NNCF](https://github.com/openvinotoolkit/nncf), which can further reduce model size and improve inference speed.

## File Structure

```
sam-3d-objects-reconstruction/
├── sam-3d-objects-reconstruction.ipynb   # Main notebook
├── sam_3d_objects_helper.py             # OV conversion & inference helper (~2500 lines)
├── gradio_helper.py                     # Gradio web UI helper
├── README.md                            # This file
├── run_e2e_test.py                      # End-to-end validation script
├── sam-3d-objects/                       # Cloned source repo (auto-downloaded)
├── models/                              # Model weights (auto-downloaded from ModelScope)
└── ov_models/                           # Converted OpenVINO IR files
    ├── dino_backbone.xml/.bin           # Merged DINOv2 ViT-L/14 (~1.2 GB)
    ├── ss_decoder.xml/.bin              # Sparse Structure decoder (~147 MB)
    ├── ss_generator.xml/.bin            # SS flow matching backbone (~1.9 GB)
    ├── slat_generator_core.xml/.bin     # SLat transformer core (~540 MB)
    ├── slat_decoder_gs.xml/.bin         # Gaussian splat decoder (~340 MB)
    ├── slat_decoder_gs_4.xml/.bin       # Gaussian splat decoder (4×) (~340 MB)
    ├── moge.xml/.bin                    # Monocular geometry estimator (~310 MB)
    └── *_proj_*.xml/.bin                # Embedder projection nets (~12 MB each)
```

## Prerequisites

- Python 3.10+
- Intel CPU (or Intel GPU with OpenVINO GPU plugin)
- ~32 GB RAM recommended (models are large)
- ~15 GB disk space (model weights + OV IR files)

### Required Packages

```bash
pip install openvino>=2024.6.0 nncf>=2.13 gradio>=4.13 \
    torch torchvision numpy Pillow matplotlib \
    omegaconf hydra-core trimesh imageio scipy einops roma \
    rootutils astor easydict lightning plyfile pyvista \
    scikit-image opencv-python igraph modelscope
```

## Quick Start

1. **Open the notebook** `sam-3d-objects-reconstruction.ipynb`
2. **Run all cells** — the notebook will:
   - Download model weights from ModelScope (~15 GB)
   - Clone the SAM-3D-Objects source code from GitHub
   - Convert all models to OpenVINO IR format
   - Run single and multi-object 3D reconstruction
3. **Interactive demo**: The Gradio section at the bottom provides a web UI for uploading custom images

## Model Details

| Model | Architecture | Parameters | OV Size | Purpose |
|-------|-------------|-----------|---------|---------|
| DINOv2 Backbone | ViT-L/14 | ~300M | ~1.2 GB | Shared image/mask encoder (2 outputs) |
| SS Generator | 24-block MOT | ~500M | ~1.9 GB | Sparse structure flow matching |
| SS Decoder | 3D ConvNet | ~37M | ~147 MB | Latent → occupancy grid |
| SLat Generator Core | 24-block Sparse Transformer | ~140M | ~540 MB | Structured latent flow matching |
| SLat Decoder (GS) | 12-block Sparse Transformer | ~85M | ~340 MB | Gaussian splat parameters |
| SLat Decoder (GS-4) | 12-block Sparse Transformer | ~85M | ~340 MB | Gaussian splat parameters (4×) |
| MoGe | ViT | ~80M | ~310 MB | Monocular geometry estimation |
| Projection Nets | LayerNorm + FFN | ~5-9M each | ~12 MB each | Feature projection |

**Total: 12 OpenVINO models, ~4 GB disk space**

## Performance Notes

- **Single object reconstruction**: ~8 minutes on CPU (25 SS steps + 25 SLat steps + decoding)
- **Typical output**: 800K–1.3M Gaussian points per object
- **Inference is deterministic** with fixed seed (`seed=42`)
- The SS and SLat generators use flow matching with 25 ODE steps each — this is the main computational bottleneck
- The pipeline uses synthetic pointmaps (bypassing MoGe's CUDA dependencies for pytorch3d) — quality is comparable for most indoor scenes

## References

- **SAM-3D-Objects**: [https://github.com/facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects)
- **OpenVINO**: [https://github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)
- **DINOv2**: [https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- **NNCF**: [https://github.com/openvinotoolkit/nncf](https://github.com/openvinotoolkit/nncf)

## License

The SAM-3D-Objects model and code are released under the [Meta License](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE). This notebook integration follows the OpenVINO Notebooks [Apache 2.0 License](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/LICENSE).
