# Copyright (c) OpenVINO Contributors
# SPDX-License-Identifier: Apache-2.0
"""
OpenVINO helper for SAM-3D-Objects (Meta) 3D reconstruction pipeline.

This module converts ALL weighted sub-models in the SAM-3D-Objects pipeline
to OpenVINO IR and wraps the full pipeline for CPU / OpenVINO inference.

Converted to OpenVINO IR:
    - DINOv2 condition embedders (ViT-L/14) — merged into 1 model with 2 outputs
    - Sparse Structure (SS) Decoder — 3D Conv decoder (latent → occupancy grid)
    - PointPatchEmbed inner attention — windowed patch attention for pointmaps
      (outer embed_pointmap_windows stays in Python; tensor weights saved to config.json)
    - SS Generator backbone — 24 MOT transformer blocks + latent mapping projections
    - SLat Generator full — t_embedder + input_blocks + 24 core transformer blocks
      + out_blocks + out_layer (ALL ~600 M weights in one OV model)
    - SLat GS Decoder — 12 sparse transformer blocks (dense, full attention replaces swin)
    - SLat GS-4 Decoder — same architecture as GS, 4 gaussians per voxel
    - SLat Mesh Decoder merged — transformer base + upsample + out_layer in single model
      (FlexiCubes mesh extraction stays in Python)
    - EmbedderFuser projection nets — per-embedder LayerNorm + FeedForward

Kept in PyTorch (CPU, not converted to OV):
    - MoGe depth model — real model loaded on CPU (mock replaced at pipeline init time)
    - SLat Decoder to_representation — coordinate-based Gaussian/Mesh construction (0 weights)
    - SparseDownsample / SparseUpsample — coordinate-only ops (0 weights)
    - Flow matching / shortcut ODE loop — pure control flow
    - Classifier-free guidance wrapper — pure control flow
"""

from __future__ import annotations

import gc
import os
import sys
import types
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import openvino as ov

# ---------------------------------------------------------------------------
# Path setup — make the sam-3d-objects project importable
# ---------------------------------------------------------------------------
_SAM3D_ROOT = Path(__file__).resolve().parents[3] / "sam-3d" / "sam-3d-objects"
_SAM3D_NOTEBOOK = _SAM3D_ROOT / "notebook"
_SAM3D_MODEL_ROOT = Path(__file__).resolve().parents[3] / "sam-3d" / "sam-3d-objects-model"

if str(_SAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM3D_ROOT))
if str(_SAM3D_NOTEBOOK) not in sys.path:
    sys.path.insert(0, str(_SAM3D_NOTEBOOK))


# ============================================================================
# 1a. CPU-compatible pytorch3d math replacements
# ============================================================================

class CPUTransform3d:
    """
    Minimal CPU replacement for ``pytorch3d.transforms.Transform3d``.

    Stores a batch of 4×4 homogeneous matrices and provides the subset of
    the pytorch3d API used by the SAM-3D-Objects pipeline:
    ``scale``, ``translate``, ``rotate``, ``compose``, ``inverse``,
    ``transform_points``, ``get_matrix``, ``to``.

    Convention (same as pytorch3d): **row-vector right-multiply**.
    Points ``P`` of shape ``(..., 3)`` are augmented to ``[P, 1]`` and
    transformed as ``P_out = [P, 1] @ M``.
    """

    def __init__(self, dtype=None, device=None, matrix=None):
        if matrix is not None:
            if not isinstance(matrix, torch.Tensor):
                matrix = torch.tensor(matrix, dtype=dtype or torch.float32)
            if matrix.dim() == 2:
                matrix = matrix.unsqueeze(0)
            self._matrix = matrix
        else:
            self._matrix = torch.eye(4, dtype=dtype or torch.float32).unsqueeze(0)
        if device is not None:
            self._matrix = self._matrix.to(device)

    # ---- builders (each returns Self for chaining) ----

    def scale(self, x, y=None, z=None):
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                # Empty tensor — treat as identity scale
                return self
            if x.dim() == 0:
                xyz = x.expand(3)
            elif x.dim() == 1 and x.shape[0] == 1:
                xyz = x.expand(3)
            elif x.dim() == 1 and x.shape[0] == 3:
                xyz = x
            elif x.dim() >= 2:
                # Batched scale: shape (..., 3)
                flat = x.reshape(-1)
                if flat.numel() >= 3:
                    xyz = flat[:3]
                elif flat.numel() == 1:
                    xyz = flat.expand(3)
                else:
                    return self
            else:
                xyz = x.reshape(-1)[:3]
        else:
            if y is None:
                y = x
            if z is None:
                z = x
            xyz = torch.tensor([x, y, z], dtype=self._matrix.dtype,
                               device=self._matrix.device)
        S = torch.zeros(1, 4, 4, dtype=self._matrix.dtype, device=self._matrix.device)
        S[0, 0, 0] = xyz[0]
        S[0, 1, 1] = xyz[1]
        S[0, 2, 2] = xyz[2]
        S[0, 3, 3] = 1.0
        self._matrix = self._matrix @ S
        return self

    def translate(self, x, y=None, z=None):
        if isinstance(x, torch.Tensor):
            xyz = x.reshape(-1)[:3]
        else:
            if y is None:
                y = 0.0
            if z is None:
                z = 0.0
            xyz = torch.tensor([x, y, z], dtype=self._matrix.dtype,
                               device=self._matrix.device)
        T = torch.eye(4, dtype=self._matrix.dtype, device=self._matrix.device).unsqueeze(0)
        T[0, 3, :3] = xyz
        self._matrix = self._matrix @ T
        return self

    def rotate(self, R):
        if isinstance(R, torch.Tensor):
            if R.dim() == 2:
                R = R.unsqueeze(0)
        else:
            R = torch.tensor(R, dtype=self._matrix.dtype,
                             device=self._matrix.device).unsqueeze(0)
        M = torch.eye(4, dtype=self._matrix.dtype, device=self._matrix.device
                       ).unsqueeze(0).expand(R.shape[0], -1, -1).clone()
        M[:, :3, :3] = R
        self._matrix = self._matrix @ M
        return self

    def compose(self, *others):
        mat = self._matrix.clone()
        for o in others:
            mat = mat @ o.get_matrix()
        return CPUTransform3d(matrix=mat)

    # ---- queries ----

    def get_matrix(self):
        return self._matrix

    def inverse(self, invert_composed=False):
        return CPUTransform3d(matrix=torch.linalg.inv(self._matrix))

    def transform_points(self, points):
        """Transform ``points`` of shape ``(..., 3)`` → ``(..., 3)``."""
        M = self._matrix.squeeze(0)          # (4, 4)
        ones = torch.ones(*points.shape[:-1], 1, dtype=points.dtype,
                          device=points.device)
        P4 = torch.cat([points, ones], dim=-1)  # (..., 4)
        out4 = P4 @ M                           # (..., 4)
        return out4[..., :3] / out4[..., 3:4].clamp(min=1e-8)

    def to(self, device_or_dtype):
        self._matrix = self._matrix.to(device_or_dtype)
        return self

    @property
    def device(self):
        return self._matrix.device

    @property
    def dtype(self):
        return self._matrix.dtype


def _cpu_look_at_view_transform(eye=None, at=None, up=None, device="cpu",
                                dist=1.0, elev=0.0, azim=0.0):
    """
    Minimal CPU replacement for ``pytorch3d.renderer.look_at_view_transform``.

    Returns ``(R, T)`` where ``R`` is ``(1, 3, 3)`` and ``T`` is ``(1, 3)``.
    """
    if eye is None:
        raise NotImplementedError("Only eye/at/up form is supported")
    eye = torch.tensor(eye, dtype=torch.float32, device=device).reshape(1, 3)
    at = torch.tensor(at, dtype=torch.float32, device=device).reshape(1, 3)
    up = torch.tensor(up, dtype=torch.float32, device=device).reshape(1, 3)

    z_axis = at - eye                                        # forward
    z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True)
    x_axis = torch.linalg.cross(up, z_axis)                  # right
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True)
    y_axis = torch.linalg.cross(z_axis, x_axis)              # corrected up

    R = torch.stack([x_axis, y_axis, z_axis], dim=1)          # (1, 3, 3)
    T = -(eye.unsqueeze(1) @ R).squeeze(1)                    # (1, 3)
    return R, T


def _cpu_quaternion_to_matrix(quaternions):
    """Convert ``(*, 4)`` quaternions (w, x, y, z) to ``(*, 3, 3)`` matrices."""
    q = quaternions
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternions of shape (*, 4), got {q.shape}")
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    B = q.shape[:-1]
    R = torch.zeros(*B, 3, 3, dtype=q.dtype, device=q.device)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _cpu_matrix_to_quaternion(matrix):
    """Convert ``(*, 3, 3)`` rotation matrices to ``(*, 4)`` quaternions (w, x, y, z)."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (*, 3, 3) matrix, got {matrix.shape}")
    B = matrix.shape[:-2]
    m = matrix
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = torch.zeros(*B, 4, dtype=matrix.dtype, device=matrix.device)
    # Use the numerically stable Shepperd method
    s = torch.sqrt(torch.clamp(t + 1, min=1e-10)) * 2  # s = 4*w
    q[..., 0] = 0.25 * s
    q[..., 1] = (m[..., 2, 1] - m[..., 1, 2]) / s
    q[..., 2] = (m[..., 0, 2] - m[..., 2, 0]) / s
    q[..., 3] = (m[..., 1, 0] - m[..., 0, 1]) / s
    # Normalize
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    return q


def _cpu_quaternion_multiply(q1, q2):
    """Hamilton product of two ``(*, 4)`` quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _cpu_quaternion_invert(q):
    """Invert a ``(*, 4)`` quaternion (w, x, y, z) → conjugate / norm²."""
    conj = q.clone()
    conj[..., 1:] = -conj[..., 1:]
    return conj / (q.norm(dim=-1, keepdim=True) ** 2).clamp(min=1e-10)


def _moduledict_get(md, key, default=None):
    """Safe ``.get()`` for ``nn.ModuleDict`` (which has no ``.get()``)."""
    if key in md:
        return md[key]
    return default


# ============================================================================
# 1.  CUDA PATCHES — remove hard CUDA dependencies for CPU / OV inference
# ============================================================================

def patch_cuda_for_cpu():
    """
    Apply patches so that the SAM-3D-Objects code can run on CPU.

    Must be called **before** importing any ``sam3d_objects`` module.
    """
    # 1. Environment hints for attention backend (avoid flash_attn / xformers)
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
    os.environ["LIDRA_SKIP_INIT"] = "true"
    # Avoid CUDA_HOME requirement
    os.environ.setdefault("CUDA_HOME", "/usr")

    # 1b. Mock CUDA-only packages that are imported at module level
    import types as _types

    _noop = lambda *a, **kw: None
    _NoopClass = type("_NoopClass", (), {
        "__init__": _noop,
        "__call__": _noop,
    })

    class _MockModule(_types.ModuleType):
        """A mock module that returns _noop for any missing attribute."""
        def __init__(self, name, attrs=None):
            super().__init__(name)
            self.__path__ = []
            self.__spec__ = None
            self.__file__ = f"<mock:{name}>"
            if attrs:
                for k, v in attrs.items():
                    setattr(self, k, v)

        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            # Check if a child mock module exists in sys.modules
            child = f"{self.__name__}.{name}"
            if child in sys.modules:
                return sys.modules[child]
            # Return a noop callable for any missing attribute (handles unknown imports)
            return _noop

    def _ensure_mock(mod_name: str, attrs: dict = None):
        """Insert a lightweight stub module if the real one is not available."""
        if mod_name in sys.modules:
            # Update existing mock with new attrs if provided
            if attrs:
                for k, v in attrs.items():
                    setattr(sys.modules[mod_name], k, v)
            return
        sys.modules[mod_name] = _MockModule(mod_name, attrs)

    # pytorch3d hierarchy ——————————————————————————————————————————————
    # Use CPU-compatible implementations instead of noops for math functions
    _pt3d_transforms_attrs = {
        "quaternion_to_matrix": _cpu_quaternion_to_matrix,
        "matrix_to_quaternion": _cpu_matrix_to_quaternion,
        "quaternion_multiply": _cpu_quaternion_multiply,
        "quaternion_invert": _cpu_quaternion_invert,
        "Transform3d": CPUTransform3d,
        "Rotate": _NoopClass,
        "Translate": _NoopClass,
        "Scale": _NoopClass,
    }
    _pt3d_structures_attrs = {
        "Meshes": _NoopClass,
        "Pointclouds": _NoopClass,
        "join_meshes_as_scene": _noop,
    }
    _pt3d_renderer_attrs = {
        "PerspectiveCameras": _NoopClass,
        "RasterizationSettings": _NoopClass,
        "MeshRasterizer": _NoopClass,
        "TexturesVertex": _NoopClass,
        "look_at_view_transform": _cpu_look_at_view_transform,
    }
    _pt3d_cameras_attrs = {
        "CamerasBase": _NoopClass,
        "PerspectiveCameras": _NoopClass,
        "camera_to_eye_at_up": _noop,
    }

    for mod, attrs in [
        ("pytorch3d", None),
        ("pytorch3d.transforms", _pt3d_transforms_attrs),
        ("pytorch3d.structures", _pt3d_structures_attrs),
        ("pytorch3d.renderer", _pt3d_renderer_attrs),
        ("pytorch3d.renderer.cameras", _pt3d_cameras_attrs),
        ("pytorch3d.renderer.camera_utils", {"camera_to_eye_at_up": _noop}),
        ("pytorch3d.renderer.mesh", None),
        ("pytorch3d.renderer.mesh.rasterizer", None),
        ("pytorch3d.renderer.mesh.shader", None),
        ("pytorch3d.renderer.mesh.textures", {"TexturesVertex": _NoopClass}),
        ("pytorch3d.io", None),
        ("pytorch3d.loss", None),
        ("pytorch3d.ops", None),
        ("pytorch3d.vis", None),
        ("pytorch3d.vis.plotly_vis", {
            "AxisArgs": type("AxisArgs", (), {"__init__": lambda self, **kw: None, "_asdict": lambda self: {}}),
            "Lighting": _NoopClass,
            "_add_camera_trace": _noop,
            "_add_pointcloud_trace": _noop,
            "_add_ray_bundle_trace": _noop,
            "_is_ray_bundle": _noop,
            "_scale_camera_to_bounds": _noop,
            "_update_axes_bounds": _noop,
        }),
        ("pytorch3d.viz", None),
        ("pytorch3d.viz.plotly_vis", None),
    ]:
        _ensure_mock(mod, attrs)

    # spconv — CUDA-only sparse convolution ——————————————————————————
    class _MockSpConvTensor:
        """Mock ``spconv.pytorch.SparseConvTensor`` that stores features/indices.

        Critical: ``features`` is a property backed by ``_features`` so that
        ``SparseTensor.replace()`` — which does ``new_data._features = feats``
        — is immediately visible through ``new_data.features``.
        """
        def __init__(self, features=None, indices=None, spatial_shape=None,
                     batch_size=None, grid=None, voxel_num=None,
                     indice_dict=None, *args, **kwargs):
            self._features = features
            self.indices = indices
            self.spatial_shape = spatial_shape if spatial_shape is not None else [64, 64, 64]
            self.batch_size = batch_size if batch_size is not None else 1
            self.grid = grid
            self.voxel_num = voxel_num or 0
            self.indice_dict = indice_dict or {}
            self.benchmark = False
            self.benchmark_record = {}
            self.thrust_allocator = None
            self._timer = None
            self.force_algo = None
            self.int8_scale = None

        @property
        def features(self):
            return self._features

        @features.setter
        def features(self, value):
            self._features = value

        def replace_feature(self, new_features):
            new = _MockSpConvTensor(
                new_features, self.indices, self.spatial_shape, self.batch_size,
                self.grid, self.voxel_num, self.indice_dict,
            )
            new.benchmark = self.benchmark
            new.benchmark_record = self.benchmark_record
            new.thrust_allocator = self.thrust_allocator
            new._timer = self._timer
            new.force_algo = self.force_algo
            new.int8_scale = self.int8_scale
            return new

    class _CenterPixelConv3d(nn.Module):
        """
        Mock sparse 3-D convolution using **center-pixel approximation**.

        Real spconv ``SubMConv3d`` applies a 3×3×3 kernel over spatial
        neighbours;  this mock only uses the *center* kernel slice
        ``weight[:, k//2, k//2, k//2, :]`` — equivalent to a pointwise
        linear transform.  Channel dimensions change correctly, so the
        U-Net encoder / decoder in the SLat generator can load checkpoint
        weights and propagate matching feature shapes.

        Weight layout (spconv convention):
            ``(out_channels, k, k, k, in_channels)``
        """
        def __init__(self, in_channels, out_channels, kernel_size,
                     stride=1, dilation=1, padding=None,
                     bias=True, indice_key=None, algo=None):
            super().__init__()
            k = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = k
            # Weight in spconv format: (out_ch, k, k, k, in_ch)
            self.weight = nn.Parameter(torch.zeros(out_channels, *k, in_channels))
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_channels))
            else:
                self.register_parameter("bias", None)

        def forward(self, x):
            """Centre-pixel pointwise convolution on ``_MockSpConvTensor``."""
            features = x.features if hasattr(x, "features") else x
            cx = self.kernel_size[0] // 2
            cy = self.kernel_size[1] // 2
            cz = self.kernel_size[2] // 2
            center_w = self.weight[:, cx, cy, cz, :]          # (out_ch, in_ch)
            # Maintain input dtype — the SLat U-Net runs in fp16
            orig_dtype = features.dtype
            out = F.linear(features.float(), center_w.float(),
                           self.bias.float() if self.bias is not None else None)
            out = out.to(orig_dtype)
            if hasattr(x, "replace_feature"):
                return x.replace_feature(out)
            return out

    _ConvAlgo = type("ConvAlgo", (), {"Native": None, "MaskImplicitGemm": None})
    _ensure_mock("spconv", None)
    _ensure_mock("spconv.pytorch", {
        "SparseConvTensor": _MockSpConvTensor,
        "SubMConv3d": _CenterPixelConv3d,
        "SparseConv3d": _CenterPixelConv3d,
        "SparseInverseConv3d": _CenterPixelConv3d,
        "ConvAlgo": _ConvAlgo,
    })

    # kaolin — CUDA-only 3D ops ——————————————————————————————————————
    for mod, attrs in [
        ("kaolin", None),
        ("kaolin.render", None),
        ("kaolin.render.mesh", None),
        ("kaolin.render.camera", {
            "Camera": _NoopClass,
            "CameraExtrinsics": _NoopClass,
            "PinholeIntrinsics": _NoopClass,
        }),
        ("kaolin.non_commercial", None),
        ("kaolin.utils", None),
        ("kaolin.utils.testing", {"check_tensor": lambda *a, **kw: True}),
        ("kaolin.visualize", {"IpyTurntableVisualizer": _NoopClass}),
    ]:
        _ensure_mock(mod, attrs)

    # Patch utils3d.numpy to have missing functions
    import utils3d.numpy as _u3d_np
    for _fn_name in ("depth_edge", "normals_edge", "points_to_normals",
                     "image_uv", "image_mesh"):
        if not hasattr(_u3d_np, _fn_name):
            setattr(_u3d_np, _fn_name, _noop)

    # moge — depth estimation model ——————————————————————————————————
    class _MockMoGeModel(nn.Module):
        """Stub MoGeModel for CPU pipeline loading."""
        def __init__(self, *a, **kw):
            super().__init__()
            self._dummy = nn.Linear(1, 1)

        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

        def infer(self, image, *a, **kw):
            """Return synthetic pointmap data matching real MoGe output format."""
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    _, H, W = image.shape
                elif image.dim() == 4:
                    _, _, H, W = image.shape
                else:
                    H, W = 518, 518
            else:
                H, W = 518, 518
            # Generate a synthetic depth surface: (H, W, 3)
            ys = torch.linspace(-0.5, 0.5, H)
            xs = torch.linspace(-0.5, 0.5, W)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            zz = torch.ones(H, W) * 2.0 + 0.1 * (xx ** 2 + yy ** 2)
            points = torch.stack([xx, yy, zz], dim=-1)  # (H, W, 3)
            return {"points": points}

    for mod, attrs in [
        ("moge", None),
        ("moge.model", None),
        ("moge.model.v1", {"MoGeModel": _MockMoGeModel}),
        ("moge.utils", None),
    ]:
        _ensure_mock(mod, attrs)

    # Restore real moge.utils geometry modules (CPU-safe, no CUDA deps).
    # These are needed by OVMoGe for recover_focal_shift post-processing.
    # The modules use relative imports, so we must set __path__ on the
    # mock moge.utils package and load dependencies in order.
    _moge_utils_dir = None
    for _sp in (
        [getattr(__import__("site"), "getusersitepackages", lambda: "")()]
        if isinstance(getattr(__import__("site"), "getusersitepackages", lambda: "")(), str)
        else getattr(__import__("site"), "getusersitepackages", lambda: [])()
    ) + getattr(__import__("site"), "getsitepackages", lambda: [])():
        _candidate = Path(_sp) / "moge" / "utils"
        if _candidate.is_dir():
            _moge_utils_dir = _candidate
            break
    if _moge_utils_dir is None:
        for _sp_path in sys.path:
            _candidate = Path(_sp_path) / "moge" / "utils"
            if _candidate.is_dir() and (_candidate / "geometry_torch.py").exists():
                _moge_utils_dir = _candidate
                break

    if _moge_utils_dir is not None:
        import importlib.util as _ilu
        # Set __path__ on mock moge and moge.utils so relative imports work
        _moge_mock = sys.modules.get("moge")
        if _moge_mock is not None:
            _moge_mock.__path__ = [str(_moge_utils_dir.parent)]
        _moge_utils_mock = sys.modules.get("moge.utils")
        if _moge_utils_mock is not None:
            _moge_utils_mock.__path__ = [str(_moge_utils_dir)]
            _moge_utils_mock.__package__ = "moge.utils"

        # Load real modules in dependency order: tools → geometry_numpy → geometry_torch
        for _mod_name, _mod_file in [
            ("moge.utils.tools", "tools.py"),
            ("moge.utils.geometry_numpy", "geometry_numpy.py"),
            ("moge.utils.geometry_torch", "geometry_torch.py"),
        ]:
            _fpath = _moge_utils_dir / _mod_file
            if _fpath.exists() and _mod_name not in sys.modules:
                _spec = _ilu.spec_from_file_location(
                    _mod_name, str(_fpath),
                    submodule_search_locations=[],
                )
                _real_mod = _ilu.module_from_spec(_spec)
                _real_mod.__package__ = "moge.utils"
                sys.modules[_mod_name] = _real_mod
                try:
                    _spec.loader.exec_module(_real_mod)
                except Exception:
                    del sys.modules[_mod_name]

    # gsplat — CUDA-only Gaussian splatting renderer ─────────────────
    _ensure_mock("gsplat", {"rasterization": _noop})

    # 2. Monkey-patch ``set_attention_backend`` so it never touches CUDA
    import sam3d_objects.pipeline.inference_pipeline as _ip_mod  # noqa: E402

    def _set_attention_backend_noop():
        os.environ["ATTN_BACKEND"] = "sdpa"
        os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"

    _ip_mod.set_attention_backend = _set_attention_backend_noop

    # 2b. Patch InferencePipeline.__init__ to avoid CUDA calls
    _orig_ip_init = _ip_mod.InferencePipeline.__init__

    def _ip_init_cpu(self, *args, **kwargs):
        kwargs["device"] = "cpu"
        # Temporarily replace torch.cuda.current_device
        _orig_current = torch.cuda.current_device
        torch.cuda.current_device = lambda: "cpu(mock)"
        try:
            _orig_ip_init(self, *args, **kwargs)
        finally:
            torch.cuda.current_device = _orig_current

    _ip_mod.InferencePipeline.__init__ = _ip_init_cpu

    # 2c. Patch load_model_from_checkpoint to use strict=False
    # (spconv stubs don't register sub-parameters so conv weights are "unexpected")
    from sam3d_objects.model import io as _io_mod

    _orig_load_ckpt = _io_mod.load_model_from_checkpoint

    def _load_ckpt_nonstrict(*args, **kwargs):
        kwargs["strict"] = False
        return _orig_load_ckpt(*args, **kwargs)

    _io_mod.load_model_from_checkpoint = _load_ckpt_nonstrict

    # Also patch the already-imported binding in inference_pipeline
    from sam3d_objects.pipeline import inference_pipeline as _ip_mod
    _ip_mod.load_model_from_checkpoint = _load_ckpt_nonstrict

    # 2d. Patch SparseFeatures2Mesh to default to CPU (FlexiCubes needs device)
    from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import SparseFeatures2Mesh as _SF2M

    _orig_sf2m_init = _SF2M.__init__

    def _sf2m_init_cpu(self, device="cpu", *args, **kwargs):
        return _orig_sf2m_init(self, device="cpu", *args, **kwargs)

    _SF2M.__init__ = _sf2m_init_cpu

    # 3. Patch DepthModel base to default to CPU
    from sam3d_objects.pipeline.depth_models.base import DepthModel  # noqa: E402

    _orig_depth_init = DepthModel.__init__

    def _depth_init_cpu(self, model, device="cpu"):
        _orig_depth_init(self, model, device="cpu")

    DepthModel.__init__ = _depth_init_cpu

    # 4. Patch Gaussian model's CUDA bias (if it exists)
    try:
        from sam3d_objects.model.backbone.tdfy_dit.utils.gaussian_model import (
            GaussianModel,
        )

        _orig_gs_init = GaussianModel.__init__

        def _gs_init_cpu(self, *a, **kw):
            _orig_gs_init(self, *a, **kw)
            # Redirect any .cuda() buffers to CPU
            for attr in ("_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"):
                val = getattr(self, attr, None)
                if val is not None and isinstance(val, torch.Tensor) and val.is_cuda:
                    setattr(self, attr, val.cpu())

        GaussianModel.__init__ = _gs_init_cpu
    except Exception:
        pass

    # 5. Patch inference_pipeline_pointmap camera_to_pytorch3d_camera default device
    try:
        import sam3d_objects.pipeline.inference_pipeline_pointmap as _ipm

        _orig_cam = _ipm.camera_to_pytorch3d_camera

        def _cam_cpu(device="cpu"):
            return _orig_cam(device="cpu")

        _ipm.camera_to_pytorch3d_camera = _cam_cpu
    except Exception:
        pass

    # 6. Patch compute_pointmap to bypass mocked pytorch3d transforms & MoGe
    #    The real compute_pointmap uses Transform3d & look_at_view_transform from
    #    pytorch3d (mocked → _NoopClass/None) and MoGe (mocked → empty dict).
    #    We replace it with a CPU-safe version that generates synthetic pointmap
    #    data so the rest of the pipeline (OV-important parts) can run.
    try:
        from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap as _IPPM

        _orig_compute_pm = _IPPM.compute_pointmap

        def _compute_pointmap_cpu(self, image, pointmap=None):
            loaded_image = self.image_to_float(image)
            loaded_image = torch.from_numpy(loaded_image)
            loaded_image = loaded_image.permute(2, 0, 1).contiguous()[:3]  # (3, H, W)
            _, H, W = loaded_image.shape

            if pointmap is not None:
                points_tensor = pointmap.to(self.device)
                if loaded_image.shape != points_tensor.shape:
                    points_tensor = torch.nn.functional.interpolate(
                        points_tensor.permute(2, 0, 1).unsqueeze(0),
                        size=(H, W), mode="nearest",
                    ).squeeze(0).permute(1, 2, 0)
                points_tensor = points_tensor.permute(2, 0, 1)  # (3, H, W)
            else:
                # Generate synthetic depth-based pointmap (bypass MoGe + pytorch3d)
                # Create a grid of (x, y) coordinates normalized to [-0.5, 0.5]
                ys = torch.linspace(-0.5, 0.5, H)
                xs = torch.linspace(-0.5, 0.5, W)
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                # Synthetic depth: slightly curved surface
                zz = torch.ones(H, W) * 2.0 + 0.1 * (xx ** 2 + yy ** 2)
                points_tensor = torch.stack([xx, yy, zz], dim=0)  # (3, H, W)

            # Clip pointmap if configured
            if hasattr(self, '_clip_pointmap') and hasattr(self, 'clip_pointmap_beyond_scale'):
                loaded_mask = self.image_to_float(image)
                loaded_mask = torch.from_numpy(loaded_mask)[..., -1] if loaded_mask.shape[-1] == 4 else torch.ones(H, W)
                points_tensor = self._clip_pointmap(points_tensor, loaded_mask)

            # Build default intrinsics (pinhole camera, focal length = image width)
            focal = float(max(H, W))
            intrinsics = torch.tensor([
                [focal, 0.0,   W / 2.0],
                [0.0,   focal, H / 2.0],
                [0.0,   0.0,   1.0],
            ], dtype=torch.float32)

            return {
                "pts_color": loaded_image,
                "pointmap": points_tensor,
                "intrinsics": intrinsics,
            }

        _IPPM.compute_pointmap = _compute_pointmap_cpu
    except Exception as _e:
        print(f"[OV-SAM3D] WARNING: could not patch compute_pointmap: {_e}")

    # Patch Gaussian model to default to CPU instead of CUDA  ————————
    try:
        from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian as _GaussianCls
        _orig_gs_init = _GaussianCls.__init__

        def _gaussian_cpu_init(self, *args, device="cpu", **kwargs):
            return _orig_gs_init(self, *args, device="cpu", **kwargs)

        def _setup_functions_cpu(self):
            if self.scaling_activation_type == "exp":
                self.scaling_activation = torch.exp
                self.inverse_scaling_activation = torch.log
            elif self.scaling_activation_type == "softplus":
                self.scaling_activation = torch.nn.functional.softplus
                from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import softplus_inverse_scaling_activation
                self.inverse_scaling_activation = softplus_inverse_scaling_activation
            from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.general_utils import inverse_sigmoid, build_scaling_rotation, strip_symmetric
            self.covariance_activation = self.build_covariance_from_scaling_rotation
            self.opacity_activation = torch.sigmoid
            self.inverse_opacity_activation = inverse_sigmoid
            self.rotation_activation = torch.nn.functional.normalize
            # Use CPU instead of .cuda()
            self.scale_bias = self.inverse_scaling_activation(torch.tensor(self.scaling_bias))
            self.rots_bias = torch.zeros((4))
            self.rots_bias[0] = 1
            self.opacity_bias = self.inverse_opacity_activation(torch.tensor(self.opacity_bias))

        _GaussianCls.__init__ = _gaussian_cpu_init
        _GaussianCls.setup_functions = _setup_functions_cpu
    except Exception:
        pass

    print("[OV-SAM3D] CUDA patches applied — running on CPU / OpenVINO")


def load_real_moge_cpu(pretrained_model_name_or_path: str = "Ruicheng/moge-vitl"):
    """
    Load the real MoGe depth estimation model on CPU.

    During ``patch_cuda_for_cpu()``, the ``moge`` package is mocked to avoid
    CUDA dependencies.  This function bypasses the mock, loads the real model
    weights from HuggingFace, and returns it on CPU in eval mode.

    Returns
    -------
    nn.Module
        The real MoGe model (``MoGeModel``), ready for ``model.infer(image)``.
    """
    import importlib
    import sys as _sys

    # Temporarily patch CUDA availability so MoGe loads on CPU
    _orig_cuda_avail = torch.cuda.is_available
    torch.cuda.is_available = lambda: False

    try:
        # Remove ALL moge modules from sys.modules so real ones can load.
        # We must remove every cached entry — not just those that look like
        # mocks — because partial real/mock mixtures break the import chain.
        _saved_moge_mods = {}
        for key in list(_sys.modules.keys()):
            if key == "moge" or key.startswith("moge."):
                _saved_moge_mods[key] = _sys.modules.pop(key)

        try:
            import moge.model.v1 as _moge_v1
            importlib.reload(_moge_v1)  # force fresh load
            MoGeModel = _moge_v1.MoGeModel
            # Verify we got the real class — the real module has a __file__
            if not hasattr(_moge_v1, "__file__") or _moge_v1.__file__ is None:
                raise ImportError("Still got mock module after removing from sys.modules")
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Cannot load real MoGe model ({e}). Please install moge: "
                "pip install MoGe@git+https://github.com/microsoft/MoGe.git"
            )

        print(f"[OV-SAM3D] Loading real MoGe model from {pretrained_model_name_or_path} …")
        model = MoGeModel.from_pretrained(pretrained_model_name_or_path)
        model = model.cpu().eval().float()
        print("[OV-SAM3D] Real MoGe model loaded on CPU")
        return model
    finally:
        torch.cuda.is_available = _orig_cuda_avail
        # Restore saved moge modules (force overwrite) so mock stays active
        # for the rest of the pipeline code that expects the mock
        _sys.modules.update(_saved_moge_mods)


def patch_pipeline_config(config):
    """
    Modify an OmegaConf pipeline config for CPU / OV inference.

    Parameters
    ----------
    config : OmegaConf DictConfig
        The loaded pipeline.yaml config.

    Returns
    -------
    config : DictConfig
        Modified in-place; also returned for convenience.
    """
    from omegaconf import OmegaConf, open_dict  # noqa: E402

    with open_dict(config):
        config.device = "cpu"
        config.compile_model = False
        config.dtype = "float32"
        # Include both Gaussian and Mesh decoding
        # Mesh decoder runs on CPU: transformer base in OV, upsample + FlexiCubes in PyTorch
        config.decode_formats = ["gaussian", "mesh"]
        # Keep rendering engine as pytorch3d (no nvdiffrast)
        config.rendering_engine = "pytorch3d"
    return config


@contextmanager
def cpu_autocast():
    """No-op context manager that replaces ``torch.autocast('cuda')``."""
    yield


# ============================================================================
# 2.  OV CONVERSION WRAPPERS — nn.Modules designed for ``ov.convert_model``
# ============================================================================

class DinoBackboneForOV(nn.Module):
    """
    Unified DINOv2 backbone wrapper for OpenVINO conversion.

    All four DINOv2 embedders (SS-image, SS-mask, SLat-image, SLat-mask) share
    identical ViT-L/14 weights.  This wrapper exports a **single** OV model
    with TWO outputs, covering both the SS stage (postnorm) and the SLat stage
    (prenorm):

        Output 0 — postnorm: ``cat(x_norm_clstoken, x_norm_patchtokens)``
        Output 1 — prenorm:  ``layer_norm(x_prenorm)``

    Input must already be 3-channel; for mask inputs the caller repeats 1→3
    channels before calling this model.

    Handles: resize → normalize → ViT forward_features → dual output.
    """

    def __init__(self, dino):
        super().__init__()
        self.backbone = dino.backbone
        self.register_buffer("mean", dino.mean.clone())
        self.register_buffer("std", dino.std.clone())
        self.resize_size = dino.resize_input_size
        self.do_normalize = dino.normalize_images

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, size=self.resize_size, mode="bilinear",
                          align_corners=False)
        if self.do_normalize:
            x = (x - self.mean) / self.std
        output = self.backbone.forward_features(x)
        # Output 0: postnorm (for SS stage, prenorm=False)
        postnorm = torch.cat(
            [output["x_norm_clstoken"].unsqueeze(1),
             output["x_norm_patchtokens"]],
            dim=1,
        )
        # Output 1: prenorm (for SLat stage, prenorm=True)
        features = output["x_prenorm"]
        prenorm = F.layer_norm(features, features.shape[-1:])
        return postnorm, prenorm


# Keep the old per-variant wrappers for backward compatibility / fallback
class DinoImageForOV(nn.Module):
    """
    Wraps a ``Dino`` embedder for 3-channel image input.

    The DINOv2 backbone is a frozen ViT-L/14 — ideal for static-graph OV export.
    Handles: resize → normalize → ViT forward → output token selection.
    """

    def __init__(self, dino):
        super().__init__()
        self.backbone = dino.backbone
        self.register_buffer("mean", dino.mean.clone())
        self.register_buffer("std", dino.std.clone())
        self.resize_size = dino.resize_input_size
        self.prenorm = dino.prenorm_features
        self.do_normalize = dino.normalize_images

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.resize_size, mode="bilinear", align_corners=False)
        if self.do_normalize:
            x = (x - self.mean) / self.std
        output = self.backbone.forward_features(x)
        if self.prenorm:
            features = output["x_prenorm"]
            return F.layer_norm(features, features.shape[-1:])
        else:
            return torch.cat(
                [output["x_norm_clstoken"].unsqueeze(1), output["x_norm_patchtokens"]],
                dim=1,
            )


class DinoMaskForOV(nn.Module):
    """
    Wraps a ``Dino`` embedder for 1-channel mask input.

    Repeats the single channel to 3 channels before feeding to ViT.
    """

    def __init__(self, dino):
        super().__init__()
        self.backbone = dino.backbone
        self.register_buffer("mean", dino.mean.clone())
        self.register_buffer("std", dino.std.clone())
        self.resize_size = dino.resize_input_size
        self.prenorm = dino.prenorm_features
        self.do_normalize = dino.normalize_images

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.resize_size, mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)  # 1-ch → 3-ch (concrete copy for OV tracing)
        if self.do_normalize:
            x = (x - self.mean) / self.std
        output = self.backbone.forward_features(x)
        if self.prenorm:
            features = output["x_prenorm"]
            return F.layer_norm(features, features.shape[-1:])
        else:
            return torch.cat(
                [output["x_norm_clstoken"].unsqueeze(1), output["x_norm_patchtokens"]],
                dim=1,
            )


class SSDecoderForOV(nn.Module):
    """
    Wraps the Sparse-Structure VAE decoder for OV conversion.

    Pure 3-D convolution decoder — no dynamic/control-flow issues.
    Input:  (B, 8, 16, 16, 16) latent volume
    Output: (B, 1, D, H, W) occupancy logits
    """

    def __init__(self, ss_decoder):
        super().__init__()
        self.input_layer = ss_decoder.input_layer
        self.middle_block = ss_decoder.middle_block
        self.blocks = ss_decoder.blocks
        self.out_layer = ss_decoder.out_layer
        self.reshape_input_to_cube = getattr(ss_decoder, "reshape_input_to_cube", False)
        if self.reshape_input_to_cube and hasattr(ss_decoder, "flat_to_cube"):
            self.flat_to_cube = ss_decoder.flat_to_cube

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reshape_input_to_cube:
            x = self.flat_to_cube(x)
        h = self.input_layer(x)
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)
        h = self.out_layer(h)
        return h


class PointPatchEmbedInnerForOV(nn.Module):
    """
    Wraps the ``inner_forward`` of ``PointPatchEmbed`` for OV conversion.

    This part runs a small windowed-attention block per patch and is
    the compute-heavy portion; ``embed_pointmap_windows`` stays in PyTorch
    because it contains NaN handling / data-dependent masking.

    Input:  (B, H, W, embed_dim) — point embeddings from embed_pointmap_windows
    Output: (B, n_windows, embed_dim) — per-window tokens
    """

    def __init__(self, ppe):
        super().__init__()
        self.patch_size = ppe.patch_size
        self.embed_dim = ppe.embed_dim
        self.cls_token = ppe.cls_token
        self.pos_embed_window = ppe.pos_embed_window
        self.pos_embed = ppe.pos_embed
        self.blocks = ppe.blocks
        self.dropout_prob = ppe.dropout_prob
        self.force_dropout_always = ppe.force_dropout_always
        if hasattr(ppe, "dropped_xyz_token"):
            self.dropped_xyz_token = ppe.dropped_xyz_token

    @torch.no_grad()
    def forward(self, x: torch.Tensor, n_h: torch.Tensor, n_w: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        H_val = int(n_h.item())
        W_val = int(n_w.item())
        ps = self.patch_size
        D = self.embed_dim
        # x comes as (B, H, W, D) where H, W are pixel dims
        x = x.view(B, H_val // ps, ps, W_val // ps, ps, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, ps * ps, D)
        cls_tok = self.cls_token.expand(x.shape[0], -1, -1)
        toks = torch.cat([cls_tok, x], dim=1)
        toks = toks + self.pos_embed_window
        for blk in self.blocks:
            toks = blk(toks)
        n_win_h = H_val // ps
        n_win_w = W_val // ps
        window_embeddings = toks[:, 0].view(B, n_win_h * n_win_w, D)
        pos_embed_patch = F.interpolate(
            self.pos_embed, size=(n_win_h, n_win_w), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1).reshape(1, n_win_h * n_win_w, D)
        out = window_embeddings + pos_embed_patch
        return out


# ---------------------------------------------------------------------------
#  Dense helpers — operate on regular tensors using weights from sparse modules
# ---------------------------------------------------------------------------

def _dense_rms_norm(rms_module, x):
    """Apply (Sparse)MultiHeadRMSNorm to a regular tensor (B, L, H, d)."""
    x_type = x.dtype
    x = x.float()
    x = F.normalize(x, dim=-1)
    return (x * rms_module.gamma * rms_module.scale).to(x_type)


def _dense_sparse_self_attn(attn, feats):
    """
    Full self-attention on flat features using weights from SparseMultiHeadAttention.

    Parameters
    ----------
    attn : SparseMultiHeadAttention  (type="self", attn_mode any)
    feats : (N, C) dense feature tensor (batch=1, flattened)

    Returns
    -------
    (N, C) output features
    """
    N, C = feats.shape
    H = attn.num_heads
    d = C // H
    qkv = attn.to_qkv(feats).reshape(N, 3, H, d)
    q, k, v = qkv.unbind(dim=1)
    if attn.qk_rms_norm:
        q = _dense_rms_norm(attn.q_rms_norm, q)
        k = _dense_rms_norm(attn.k_rms_norm, k)
    h = F.scaled_dot_product_attention(
        q.unsqueeze(0).transpose(1, 2),
        k.unsqueeze(0).transpose(1, 2),
        v.unsqueeze(0).transpose(1, 2),
    )
    h = h.transpose(1, 2).squeeze(0).reshape(N, C)
    return attn.to_out(h)


def _dense_sparse_cross_attn(attn, feats, context):
    """
    Cross-attention on flat features using weights from SparseMultiHeadAttention.

    Parameters
    ----------
    attn : SparseMultiHeadAttention  (type="cross", attn_mode="full")
    feats : (N, C) query features
    context : (1, M, C_ctx) key-value context

    Returns
    -------
    (N, C) output features
    """
    N = feats.shape[0]
    C = attn.channels
    H = attn.num_heads
    d = C // H
    q = attn.to_q(feats).reshape(N, H, d)
    ctx = context.squeeze(0) if context.dim() == 3 else context
    kv = attn.to_kv(ctx)
    M = kv.shape[0]
    kv = kv.reshape(M, 2, H, d)
    k, v = kv.unbind(dim=1)
    if attn.qk_rms_norm:
        q = _dense_rms_norm(attn.q_rms_norm, q)
        k = _dense_rms_norm(attn.k_rms_norm, k)
    h = F.scaled_dot_product_attention(
        q.unsqueeze(0).transpose(1, 2),
        k.unsqueeze(0).transpose(1, 2),
        v.unsqueeze(0).transpose(1, 2),
    )
    h = h.transpose(1, 2).squeeze(0).reshape(N, C)
    return attn.to_out(h)


def _dense_sparse_ffn(ffn, feats):
    """Apply SparseFeedForwardNet to a regular tensor using F.linear."""
    for layer in ffn.mlp:
        # SparseLinear inherits nn.Linear — use its weight/bias directly
        if hasattr(layer, "weight") and hasattr(layer, "bias"):
            feats = F.linear(feats, layer.weight, layer.bias)
        else:
            # SparseGELU or other activation — call on raw tensor
            feats = F.gelu(feats, approximate="tanh")
    return feats


class SSGeneratorForOV(nn.Module):
    """
    Wraps the entire SS Generator backbone for OV conversion.

    Includes: TimestepEmbedder, adaLN modulation, 24 MOTModulatedTransformerCrossBlock,
    and all LatentMapping projections (to_input / to_output / pos_emb).

    The original dict-based MOT architecture is unrolled into explicit
    shape / pose tensor operations — no ``_pytree`` or ``dict`` in the forward.
    """

    def __init__(self, wrapper_model):
        super().__init__()
        self.t_embedder = wrapper_model.t_embedder
        self.d_embedder = getattr(wrapper_model, "d_embedder", None)
        self.share_mod = wrapper_model.share_mod
        if self.share_mod:
            self.adaLN_modulation = wrapper_model.adaLN_modulation
        self.blocks = wrapper_model.blocks
        self.latent_mapping = wrapper_model.latent_mapping
        self.dtype = wrapper_model.dtype

        # Dynamically find the merged group (any key != "shape")
        pose_names = []
        self._merged_key = None
        for merged, names in wrapper_model.latent_share_transformer.items():
            if merged != "shape":
                self._merged_key = merged
                pose_names = list(names)
                break
        self._pose_names = pose_names
        self._pose_lengths = [
            wrapper_model.latent_mapping[n].pos_emb.shape[0] for n in pose_names
        ]

    @torch.no_grad()
    def forward(
        self,
        shape_latent: torch.Tensor,
        pose_0: torch.Tensor,
        pose_1: torch.Tensor,
        pose_2: torch.Tensor,
        pose_3: torch.Tensor,
        t: torch.Tensor,
        d: torch.Tensor,
        cond: torch.Tensor,
    ):
        """
        Parameters
        ----------
        shape_latent : (B, 4096, 8)
        pose_0..3 : (B, 1, in_ch_i) — one per pose sub-latent
        t, d : (B,) timestep / shortcut step
        cond : (B, M, C_cond) condition tokens
        """
        # ── Project inputs ─────────────────────────────────────────────
        pose_tensors = [pose_0, pose_1, pose_2, pose_3]
        shape = self.latent_mapping["shape"].to_input(shape_latent)
        parts = []
        for name, raw in zip(self._pose_names, pose_tensors):
            parts.append(self.latent_mapping[name].to_input(raw))
        pose = torch.cat(parts, dim=1)

        # ── Timestep embedding ─────────────────────────────────────────
        t_emb = self.t_embedder(t)
        if self.d_embedder is not None:
            t_emb = t_emb + self.d_embedder(d)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)

        # ── Cast ───────────────────────────────────────────────────────
        shape = shape.type(self.dtype)
        pose = pose.type(self.dtype)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        # ── 24 MOT blocks (unrolled from dict) ────────────────────────
        for block in self.blocks:
            shape, pose = self._mot_block(block, shape, pose, t_emb, cond)

        shape = shape.float()
        pose = pose.float()

        # ── Split pose & project outputs ───────────────────────────────
        shape_out = self.latent_mapping["shape"].to_output(shape)
        pose_outs = []
        idx = 0
        for name, length in zip(self._pose_names, self._pose_lengths):
            pose_outs.append(
                self.latent_mapping[name].to_output(pose[:, idx : idx + length])
            )
            idx += length
        return (shape_out, *pose_outs)

    # ------------------------------------------------------------------ helpers
    def _mot_block(self, block, shape, pose, t_emb, cond):
        """Single MOTModulatedTransformerCrossBlock — explicit shape/pose ops."""
        pk = self._merged_key  # dynamic key (e.g. '6drotation_normalized')
        mod = t_emb
        if block.share_mod:
            s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = mod.chunk(6, dim=1)
        else:
            s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = block.adaLN_modulation(mod).chunk(6, dim=1)

        # ── Self-attention ──
        h_s = block.norm1["shape"](shape) * (1 + sc_msa.unsqueeze(1)) + s_msa.unsqueeze(1)
        h_p = block.norm1[pk](pose) * (1 + sc_msa.unsqueeze(1)) + s_msa.unsqueeze(1)
        h_s, h_p = self._mot_self_attn(block.self_attn, h_s, h_p)
        shape = shape + h_s * g_msa.unsqueeze(1)
        pose = pose + h_p * g_msa.unsqueeze(1)

        # ── Cross-attention ──
        shape = shape + block.cross_attn["shape"](block.norm2["shape"](shape), cond)
        pose = pose + block.cross_attn[pk](block.norm2[pk](pose), cond)

        # ── FFN ──
        h_s = block.norm3["shape"](shape) * (1 + sc_mlp.unsqueeze(1)) + s_mlp.unsqueeze(1)
        h_p = block.norm3[pk](pose) * (1 + sc_mlp.unsqueeze(1)) + s_mlp.unsqueeze(1)
        shape = shape + block.mlp["shape"](h_s) * g_mlp.unsqueeze(1)
        pose = pose + block.mlp[pk](h_p) * g_mlp.unsqueeze(1)
        return shape, pose

    def _mot_self_attn(self, attn, h_s, h_p):
        """MOTMultiHeadSelfAttention — per-latent QKV, multi-modal attention."""
        pk = self._merged_key
        B = h_s.shape[0]
        H, d = attn.num_heads, attn.head_dim

        qkv_s = attn.to_qkv["shape"](h_s).reshape(B, -1, 3, H, d)
        qkv_p = attn.to_qkv[pk](h_p).reshape(B, -1, 3, H, d)
        q_s, k_s, v_s = qkv_s.unbind(dim=2)
        q_p, k_p, v_p = qkv_p.unbind(dim=2)

        if attn.qk_rms_norm:
            q_s = _dense_rms_norm(attn.q_rms_norm["shape"], q_s)
            k_s = _dense_rms_norm(attn.k_rms_norm["shape"], k_s)
            q_p = _dense_rms_norm(attn.q_rms_norm[pk], q_p)
            k_p = _dense_rms_norm(attn.k_rms_norm[pk], k_p)

        # Protected modality (shape) self-attends only
        out_s = F.scaled_dot_product_attention(
            q_s.transpose(1, 2), k_s.transpose(1, 2), v_s.transpose(1, 2),
        ).transpose(1, 2).reshape(B, -1, H * d)

        # Pose attends to [pose, shape(no-grad)]
        k_all = torch.cat([k_p, k_s], dim=1)
        v_all = torch.cat([v_p, v_s], dim=1)
        out_p = F.scaled_dot_product_attention(
            q_p.transpose(1, 2), k_all.transpose(1, 2), v_all.transpose(1, 2),
        ).transpose(1, 2).reshape(B, -1, H * d)

        return attn.to_out["shape"](out_s), attn.to_out[pk](out_p)


class SLatGeneratorCoreForOV(nn.Module):
    """
    Dense-equivalent of the SLat Generator's main transformer loop.

    Converts 24 ``ModulatedSparseTransformerCrossBlock`` (full-attention mode)
    to operate on flat (N, C) feature tensors without SparseTensor.

    **Not included** (kept in Python wrapper):
      * input/output ``SparseResBlock3d`` (spconv-dependent)
      * SparseTensor construction / coordinate management

    Input:  feats (N, C_model), coords_xyz (N, 3),
            t_emb (1, 6*C) if share_mod else (1, C),
            cond (1, M, C_cond)
    Output: feats_out (N, C_model)
    """

    def __init__(self, slat_gen):
        super().__init__()
        self.pos_embedder = slat_gen.pos_embedder
        self.blocks = slat_gen.blocks        # 24 ModulatedSparseTransformerCrossBlock
        self.share_mod = slat_gen.share_mod
        self.dtype = slat_gen.dtype

    @torch.no_grad()
    def forward(self, feats, coords_xyz, t_emb, cond):
        """
        Parameters
        ----------
        feats : (N, C_model) — post input-block features
        coords_xyz : (N, 3) — integer voxel coordinates
        t_emb : (1, 6*C_model) if share_mod else (1, C_model) — timestep embedding
        cond : (1, M, C_cond) — condition tokens
        """
        h = feats + self.pos_embedder(coords_xyz).type(feats.dtype)

        for block in self.blocks:
            h = self._sparse_cross_block(block, h, t_emb, cond)
        return h

    def _sparse_cross_block(self, block, x, mod, context):
        """ModulatedSparseTransformerCrossBlock — dense feats, full attention."""
        if block.share_mod:
            s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = mod.chunk(6, dim=-1)
        else:
            s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = block.adaLN_modulation(mod).chunk(6, dim=-1)

        # Squeeze batch dim for broadcast (mod is (1, 6C) or (6C,))
        def _sq(t):
            return t.squeeze(0) if t.dim() > 1 else t

        # ── self-attention ──
        h = block.norm1(x) * (1 + _sq(sc_msa)) + _sq(s_msa)
        h = _dense_sparse_self_attn(block.self_attn, h)
        x = x + h * _sq(g_msa)

        # ── cross-attention ──
        h = block.norm2(x)
        h = _dense_sparse_cross_attn(block.cross_attn, h, context)
        x = x + h

        # ── FFN ──
        h = block.norm3(x) * (1 + _sq(sc_mlp)) + _sq(s_mlp)
        h = _dense_sparse_ffn(block.mlp, h)
        x = x + h * _sq(g_mlp)
        return x


class DenseResBlock(nn.Module):
    """
    Dense-equivalent of ``SparseResBlock3d`` for OV export.

    Re-implements the forward path using only standard PyTorch ops:
    - ``_CenterPixelConv3d``  →  ``nn.Linear`` (center-pixel slice)
    - ``SparseLinear``        →  ``nn.Linear``
    - ``LayerNorm32``         → kept as-is (already works on plain tensors)
    - ``emb_layers`` (SiLU → Linear)  → kept as-is

    SparseDownsample / SparseUpsample are **not** included
    (they have 0 learnable weights and operate on coordinates only).
    """

    def __init__(self, sparse_block):
        super().__init__()
        self.norm1 = sparse_block.norm1
        self.norm2 = sparse_block.norm2
        self.emb_layers = sparse_block.emb_layers

        # Extract center-pixel weights from conv1
        # SparseConv3d wraps the mocked _CenterPixelConv3d as .conv
        conv1_inner = self._get_inner_conv(sparse_block.conv1)
        k = conv1_inner.kernel_size
        cx, cy, cz = k[0] // 2, k[1] // 2, k[2] // 2
        cw1 = conv1_inner.weight[:, cx, cy, cz, :].data.clone()
        self.conv1 = nn.Linear(cw1.shape[1], cw1.shape[0],
                               bias=conv1_inner.bias is not None)
        self.conv1.weight.data.copy_(cw1)
        if conv1_inner.bias is not None:
            self.conv1.bias.data.copy_(conv1_inner.bias.data)

        # Extract center-pixel weights from conv2
        conv2_inner = self._get_inner_conv(sparse_block.conv2)
        k = conv2_inner.kernel_size
        cx, cy, cz = k[0] // 2, k[1] // 2, k[2] // 2
        cw2 = conv2_inner.weight[:, cx, cy, cz, :].data.clone()
        self.conv2 = nn.Linear(cw2.shape[1], cw2.shape[0],
                               bias=conv2_inner.bias is not None)
        self.conv2.weight.data.copy_(cw2)
        if conv2_inner.bias is not None:
            self.conv2.bias.data.copy_(conv2_inner.bias.data)

        # Skip connection: Identity or SparseLinear (nn.Linear)
        self.has_skip = not isinstance(sparse_block.skip_connection, nn.Identity)
        if self.has_skip:
            sk = sparse_block.skip_connection
            self.skip = nn.Linear(sk.in_features, sk.out_features,
                                  bias=sk.bias is not None)
            self.skip.weight.data.copy_(sk.weight.data)
            if sk.bias is not None:
                self.skip.bias.data.copy_(sk.bias.data)

    @staticmethod
    def _get_inner_conv(conv_module):
        """Get the inner _CenterPixelConv3d from SparseConv3d wrapper or direct module."""
        if hasattr(conv_module, 'conv'):
            return conv_module.conv  # sam3d_objects SparseConv3d wraps as .conv
        return conv_module  # direct _CenterPixelConv3d

    def forward(self, feats, t_emb):
        """
        Parameters
        ----------
        feats : (N, C_in) — per-voxel features
        t_emb : (B, C_t) — timestep embedding
        """
        emb_out = self.emb_layers(t_emb)
        scale, shift = emb_out.chunk(2, dim=-1)

        h = self.norm1(feats)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h) * (1 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)

        if self.has_skip:
            skip = self.skip(feats)
        else:
            skip = feats
        return h + skip


class SLatGeneratorFullForOV(nn.Module):
    """
    Dense-equivalent of the **full** SLat Generator backbone,
    including all U-Net components:

    ``t_embedder → input_layer → input_blocks → 24 core transformer blocks
    → out_blocks → out_layer``

    This replaces ``SLatGeneratorCoreForOV`` and eliminates all remaining
    PyTorch weight dependencies in the SLat Generator (~46M additional params).

    SparseDownsample / SparseUpsample (0 weights) are removed — in mock mode
    (centre-pixel ``_CenterPixelConv3d``) they have no effect on features.

    Input:  feats (N, C_in=8), coords_xyz (N, 3), t (B,), cond (1, M, C_cond)
    Output: feats_out (N, C_in=8)
    """

    def __init__(self, backbone):
        super().__init__()
        self.t_embedder = backbone.t_embedder
        self.pos_embedder = backbone.pos_embedder
        self.share_mod = backbone.share_mod
        self.use_skip_connection = backbone.use_skip_connection
        self.dtype = backbone.dtype

        # d_embedder (may not exist)
        self.has_d_embedder = (
            hasattr(backbone, "d_embedder")
            and backbone.d_embedder is not None
        )
        if self.has_d_embedder:
            self.d_embedder = backbone.d_embedder

        # adaLN_modulation for share_mod mode
        if self.share_mod:
            self.adaLN_modulation = backbone.adaLN_modulation

        # input/output layers (SparseLinear = nn.Linear)
        il = backbone.input_layer
        self.input_layer = nn.Linear(
            il.in_features, il.out_features, bias=il.bias is not None
        )
        self.input_layer.weight.data.copy_(il.weight.data)
        if il.bias is not None:
            self.input_layer.bias.data.copy_(il.bias.data)

        ol = backbone.out_layer
        self.out_layer = nn.Linear(
            ol.in_features, ol.out_features, bias=ol.bias is not None
        )
        self.out_layer.weight.data.copy_(ol.weight.data)
        if ol.bias is not None:
            self.out_layer.bias.data.copy_(ol.bias.data)

        # Dense-equivalent input/output ResBlocks
        self.in_res_blocks = nn.ModuleList(
            [DenseResBlock(b) for b in backbone.input_blocks]
        )
        self.out_res_blocks = nn.ModuleList(
            [DenseResBlock(b) for b in backbone.out_blocks]
        )

        # 24 core transformer blocks
        self.core_blocks = backbone.blocks

    @torch.no_grad()
    def forward(self, feats, coords_xyz, t, cond):
        """
        Parameters
        ----------
        feats      : (N, 8)  — per-voxel input latent
        coords_xyz : (N, 3)  — integer voxel coordinates
        t          : (B,)    — timestep
        cond       : (1, M, C_cond) — condition tokens
        """
        # ── Timestep embedding ──
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(feats.dtype)
        cond = cond.type(feats.dtype)

        # ── Input layer ──
        h = self.input_layer(feats)

        # ── Input ResBlocks (skip-save) ──
        skips = []
        for block in self.in_res_blocks:
            h = block(h, t_emb)
            skips.append(h)

        # ── Position embedding + core transformer ──
        h = h + self.pos_embedder(coords_xyz).type(h.dtype)
        for block in self.core_blocks:
            h = self._sparse_cross_block(block, h, t_emb, cond)

        # ── Output ResBlocks (with skip connections) ──
        for block, skip in zip(self.out_res_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(torch.cat([h, skip], dim=-1), t_emb)
            else:
                h = block(h, t_emb)

        # ── Final norm + output layer ──
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        return h

    def _sparse_cross_block(self, block, x, mod, context):
        """ModulatedSparseTransformerCrossBlock — dense feats, full attention."""
        if block.share_mod:
            s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = mod.chunk(6, dim=-1)
        else:
            s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = (
                block.adaLN_modulation(mod).chunk(6, dim=-1)
            )

        def _sq(t):
            return t.squeeze(0) if t.dim() > 1 else t

        # ── self-attention ──
        h = block.norm1(x) * (1 + _sq(sc_msa)) + _sq(s_msa)
        h = _dense_sparse_self_attn(block.self_attn, h)
        x = x + h * _sq(g_msa)

        # ── cross-attention ──
        h = block.norm2(x)
        h = _dense_sparse_cross_attn(block.cross_attn, h, context)
        x = x + h

        # ── FFN ──
        h = block.norm3(x) * (1 + _sq(sc_mlp)) + _sq(s_mlp)
        h = _dense_sparse_ffn(block.mlp, h)
        x = x + h * _sq(g_mlp)
        return x


class SLatDecoderForOV(nn.Module):
    """
    Dense-equivalent of ``SparseTransformerBase`` + output layer for
    SLat GS / GS-4 / Mesh decoders.

    Uses **full attention** in place of swin-windowed attention.
    All weights are identical — only the attention scope changes.

    Input:  feats (N, latent_channels), coords_xyz (N, 3)
    Output: out_feats (N, out_channels)
    """

    def __init__(self, decoder_base):
        super().__init__()
        # SparseLinear inherits nn.Linear — weights are stored as .weight/.bias
        self.input_layer = decoder_base.input_layer
        self.pos_embedder = decoder_base.pos_embedder
        self.blocks = decoder_base.blocks        # 12 SparseTransformerBlock
        self.out_layer = decoder_base.out_layer
        self.pe_mode = decoder_base.pe_mode
        self.dtype = decoder_base.dtype

    @torch.no_grad()
    def forward(self, feats, coords_xyz):
        """
        Parameters
        ----------
        feats : (N, latent_channels) — per-voxel latent features
        coords_xyz : (N, 3) — integer voxel coordinates
        """
        # input_layer is SparseLinear (nn.Linear) — call via F.linear
        h = F.linear(feats, self.input_layer.weight, self.input_layer.bias)
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(coords_xyz)
        h = h.type(self.dtype)

        for block in self.blocks:
            h = self._sparse_block(block, h)

        # layer_norm + out_layer
        h = F.layer_norm(h, h.shape[-1:])
        h = F.linear(h, self.out_layer.weight, self.out_layer.bias)
        return h

    def _sparse_block(self, block, x):
        """SparseTransformerBlock — dense feats, full attention."""
        h = block.norm1(x)
        h = _dense_sparse_self_attn(block.attn, h)
        x = x + h
        h = block.norm2(x)
        h = _dense_sparse_ffn(block.mlp, h)
        x = x + h
        return x


class SLatMeshDecoderBaseForOV(nn.Module):
    """
    Dense-equivalent of ``SparseTransformerBase`` (transformer base only)
    for the mesh decoder.

    Unlike ``SLatDecoderForOV`` this does **not** include the ``out_layer``.
    The mesh decoder's out_layer sits *after* the upsample blocks
    (SparseSubdivide + SparseConv3d) which cannot be converted to OV.

    Input:  feats (N, latent_channels), coords_xyz (N, 3)
    Output: base_feats (N, model_channels) — after blocks + norm
    """

    def __init__(self, decoder_base):
        super().__init__()
        self.input_layer = decoder_base.input_layer
        self.pos_embedder = decoder_base.pos_embedder
        self.blocks = decoder_base.blocks
        self.pe_mode = decoder_base.pe_mode
        self.dtype = decoder_base.dtype

    @torch.no_grad()
    def forward(self, feats, coords_xyz):
        h = F.linear(feats, self.input_layer.weight, self.input_layer.bias)
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(coords_xyz)
        h = h.type(self.dtype)
        for block in self.blocks:
            h = self._sparse_block(block, h)
        h = F.layer_norm(h, h.shape[-1:])
        return h

    def _sparse_block(self, block, x):
        """SparseTransformerBlock — dense feats, full attention."""
        h = block.norm1(x)
        h = _dense_sparse_self_attn(block.attn, h)
        x = x + h
        h = block.norm2(x)
        h = _dense_sparse_ffn(block.mlp, h)
        x = x + h
        return x


class MeshDecoderUpsampleForOV(nn.Module):
    """
    Dense-equivalent of the mesh decoder's **upsample blocks + out_layer**,
    merged into a single OV-exportable model.

    Replaces:
    - ``SparseSubdivideBlock3d × 2``  (5.89 M params)
    - ``SparseLinear`` out_layer       (9 797 params)

    Implementation notes:
    - ``SparseGroupNorm32`` → ``nn.GroupNorm`` on ``(1, C, N)`` tensors.
    - ``SparseConv3d``  (center-pixel ``_CenterPixelConv3d``) → ``nn.Linear``.
    - ``SparseSubdivide`` (0 weights) → ``torch.repeat_interleave(…, 8, dim=0)``.
    - ``SparseLinear`` → ``nn.Linear``.

    Input:  feats  ``(N, 768)``  — per-voxel features from transformer base
    Output: out    ``(64N, 101)`` — upsampled & projected features
    """

    def __init__(self, mesh_decoder):
        super().__init__()

        up0 = mesh_decoder.upsample[0]
        up1 = mesh_decoder.upsample[1]

        # ── Upsample block 0:  768 → 192 ──────────────────────────────
        self.norm0_pre = self._copy_group_norm(up0.act_layers[0])
        self.conv0_1 = self._center_pixel_to_linear(up0.out_layers[0])
        self.norm0_post = self._copy_group_norm(up0.out_layers[1])
        self.conv0_2 = self._center_pixel_to_linear(up0.out_layers[3])
        self.skip0 = self._center_pixel_to_linear(up0.skip_connection)

        # ── Upsample block 1:  192 → 96 ───────────────────────────────
        self.norm1_pre = self._copy_group_norm(up1.act_layers[0])
        self.conv1_1 = self._center_pixel_to_linear(up1.out_layers[0])
        self.norm1_post = self._copy_group_norm(up1.out_layers[1])
        self.conv1_2 = self._center_pixel_to_linear(up1.out_layers[3])
        self.skip1 = self._center_pixel_to_linear(up1.skip_connection)

        # ── Out layer: 96 → 101 ───────────────────────────────────────
        ol = mesh_decoder.out_layer
        self.out_layer = nn.Linear(ol.in_features, ol.out_features,
                                   bias=ol.bias is not None)
        self.out_layer.weight.data.copy_(ol.weight.data)
        if ol.bias is not None:
            self.out_layer.bias.data.copy_(ol.bias.data)

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _copy_group_norm(sparse_gn):
        """Clone weight/bias from ``SparseGroupNorm32`` into plain ``nn.GroupNorm``."""
        gn = nn.GroupNorm(sparse_gn.num_groups, sparse_gn.num_channels)
        gn.weight.data.copy_(sparse_gn.weight.data)
        gn.bias.data.copy_(sparse_gn.bias.data)
        return gn

    @staticmethod
    def _center_pixel_to_linear(conv_module):
        """Extract center-pixel slice from ``_CenterPixelConv3d`` → ``nn.Linear``."""
        inner = conv_module.conv if hasattr(conv_module, "conv") else conv_module
        k = inner.kernel_size
        cx, cy, cz = k[0] // 2, k[1] // 2, k[2] // 2
        cw = inner.weight[:, cx, cy, cz, :].data.clone()
        lin = nn.Linear(cw.shape[1], cw.shape[0], bias=inner.bias is not None)
        lin.weight.data.copy_(cw)
        if inner.bias is not None:
            lin.bias.data.copy_(inner.bias.data)
        return lin

    def _group_norm(self, norm, feats):
        """``(N, C)`` → ``(1, C, N)`` → GroupNorm → ``(N, C)``."""
        h = feats.unsqueeze(0).permute(0, 2, 1)
        h = norm(h)
        return h.permute(0, 2, 1).squeeze(0)

    def _subdivide_block(self, feats, norm_pre, conv1, norm_post, conv2, skip):
        """One ``SparseSubdivideBlock3d`` forward with feature expand."""
        h = F.silu(self._group_norm(norm_pre, feats))
        # Subdivide feature duplication (8 sub-voxels per voxel)
        h = torch.repeat_interleave(h, 8, dim=0)
        x = torch.repeat_interleave(feats, 8, dim=0)
        h = conv1(h)
        h = F.silu(self._group_norm(norm_post, h))
        h = conv2(h)
        return h + skip(x)

    @torch.no_grad()
    def forward(self, feats):
        """
        Parameters
        ----------
        feats : ``(N, 768)``

        Returns
        -------
        ``(64N, 101)``
        """
        h = self._subdivide_block(
            feats, self.norm0_pre, self.conv0_1, self.norm0_post,
            self.conv0_2, self.skip0,
        )
        h = self._subdivide_block(
            h, self.norm1_pre, self.conv1_1, self.norm1_post,
            self.conv1_2, self.skip1,
        )
        return self.out_layer(h)


class SLatMeshDecoderMergedForOV(nn.Module):
    """
    Merged wrapper that combines ``SLatMeshDecoderBaseForOV`` (transformer
    blocks) and ``MeshDecoderUpsampleForOV`` (upsample + out_layer) into a
    single OV-exportable model.

    This eliminates one numpy↔torch round-trip between the two stages.

    Input:  feats (N, latent_channels), coords_xyz (N, 3)
    Output: out_feats (64N, out_channels)
    """

    def __init__(self, decoder_model):
        super().__init__()
        self.base = SLatMeshDecoderBaseForOV(decoder_model)
        self.upsample = MeshDecoderUpsampleForOV(decoder_model)

    @torch.no_grad()
    def forward(self, feats, coords_xyz):
        base_feats = self.base(feats, coords_xyz)
        return self.upsample(base_feats)


class MoGeForOV(nn.Module):
    """
    Wraps the MoGe depth-estimation model for OV conversion.

    Extracts the backbone + head computation into a clean forward path
    that returns ``(points, mask)`` tensors (no dict, no torch.autocast).

    The caller (OVMoGe) is responsible for:
      - resizing the input image to the num_tokens-determined resolution
      - post-processing (recover_focal_shift, mask application)
    """

    def __init__(self, moge_model):
        super().__init__()
        self.backbone = moge_model.backbone
        self.head = moge_model.head
        # Copy registered buffers
        self.image_mean = moge_model.image_mean
        self.image_std = moge_model.image_std
        self._intermediate_layers = moge_model.intermediate_layers
        # Store remap config
        self.remap_output = getattr(moge_model, "remap_output", "linear")

    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        """Apply nonlinear remapping matching MoGeModel._remap_points."""
        if self.remap_output == "linear":
            pass
        elif self.remap_output == "sinh":
            points = torch.sinh(points)
        elif self.remap_output == "exp":
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output == "sinh_exp":
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        return points

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        image : (1, 3, H, W)  float32 in [0, 1], already resized to target
                resolution (num_tokens-based).

        Returns
        -------
        points : (1, H, W, 3)  predicted 3-D point-map after _remap_points
        mask   : (1, H, W)     confidence mask (logits, not sigmoid)
        """
        original_height, original_width = image.shape[-2:]

        # Normalize for DINOv2
        image_norm = (image - self.image_mean) / self.image_std
        # Pad to 14-divisible
        H14 = (original_height // 14) * 14
        W14 = (original_width // 14) * 14
        image_14 = F.interpolate(
            image_norm, (H14, W14), mode="bilinear",
            align_corners=False, antialias=True,
        )

        # Get backbone features
        features = self.backbone.get_intermediate_layers(
            image_14, self._intermediate_layers, return_class_token=True,
        )

        # Head prediction — head receives the normalized (non-14-padded) image
        points, mask = self.head(features, image_norm)

        # Resize to input resolution
        points = F.interpolate(
            points, (original_height, original_width),
            mode="bilinear", align_corners=False, antialias=False,
        )
        mask = F.interpolate(
            mask, (original_height, original_width),
            mode="bilinear", align_corners=False, antialias=False,
        )
        points = points.permute(0, 2, 3, 1)  # (1, H, W, 3)
        mask = mask.squeeze(1)                 # (1, H, W)

        # Apply nonlinear remapping (critical for 'exp' mode)
        points = self._remap_points(points)

        return points, mask


class PointProjForOV(nn.Module):
    """
    Wraps the PointPatchEmbed outer-path learnable ops for OV conversion.

    Converts ``point_proj`` (nn.Linear) and ``invalid_xyz_token``
    application into a single OV model.

    Input:  (B, H, W, 3)  — remapped XYZ coordinates
            (B, H, W)     — valid_mask (True = valid point)
    Output: (B, H, W, embed_dim)  — projected point embeddings
    """

    def __init__(self, ppe_model):
        super().__init__()
        self.point_proj = ppe_model.point_proj
        self.invalid_xyz_token = ppe_model.invalid_xyz_token

    @torch.no_grad()
    def forward(
        self,
        xyz_remapped: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.point_proj(xyz_remapped)            # (B, H, W, D)
        inv_mask = ~valid_mask.bool()
        x = torch.where(
            inv_mask.unsqueeze(-1),
            self.invalid_xyz_token.expand_as(x),
            x,
        )
        return x


class EmbedderProjectionForOV(nn.Module):
    """
    Wraps a single EmbedderFuser projection net (LayerNorm → FeedForward)
    for OV conversion.

    Input:  (B, L, embed_dim)   — raw embedder output
    Output: (B, L, output_dim)  — projected tokens
    """

    def __init__(self, projection_net):
        super().__init__()
        self.net = projection_net  # nn.Sequential(LayerNorm, FeedForward)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# 3.  CONVERSION FUNCTIONS — convert PyTorch models to OpenVINO IR
# ============================================================================

def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def convert_dino_image(
    dino_model,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 3, 518, 518),
) -> ov.Model:
    """Convert a 3-channel DINOv2 image embedder to OpenVINO IR."""
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[OV-SAM3D] Skipping {output_path} — already exists")
        return ov.Core().read_model(str(output_path))

    wrapper = DinoImageForOV(dino_model).eval().float()
    example_input = torch.randn(input_shape, dtype=torch.float32)
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example_input)
    ov.save_model(ov_model, str(output_path))
    print(f"[OV-SAM3D] Saved DINOv2 image model → {output_path}")
    del wrapper
    _cleanup()
    return ov_model


def convert_dino_mask(
    dino_model,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 1, 518, 518),
) -> ov.Model:
    """Convert a 1-channel DINOv2 mask embedder to OpenVINO IR."""
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[OV-SAM3D] Skipping {output_path} — already exists")
        return ov.Core().read_model(str(output_path))

    wrapper = DinoMaskForOV(dino_model).eval().float()
    example_input = torch.randn(input_shape, dtype=torch.float32)
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example_input)
    ov.save_model(ov_model, str(output_path))
    print(f"[OV-SAM3D] Saved DINOv2 mask model → {output_path}")
    del wrapper
    _cleanup()
    return ov_model


def convert_dino_backbone(
    dino_model,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 3, 518, 518),
) -> ov.Model:
    """
    Convert the **shared** DINOv2 backbone to a single OpenVINO IR with
    two outputs (postnorm, prenorm).

    All four DINOv2 embedders share identical ViT-L/14 weights.
    This function exports ONE model (~1.2 GB) instead of four copies.
    """
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[OV-SAM3D] Skipping {output_path} — already exists")
        return ov.Core().read_model(str(output_path))

    wrapper = DinoBackboneForOV(dino_model).eval().float()
    example_input = torch.randn(input_shape, dtype=torch.float32)
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example_input)
    ov.save_model(ov_model, str(output_path))
    print(f"[OV-SAM3D] Saved merged DINOv2 backbone → {output_path}")
    del wrapper
    _cleanup()
    return ov_model


def convert_ss_decoder(
    ss_decoder,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 8, 16, 16, 16),
) -> ov.Model:
    """Convert the SS decoder (3D-Conv VAE decoder) to OpenVINO IR."""
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[OV-SAM3D] Skipping {output_path} — already exists")
        return ov.Core().read_model(str(output_path))

    wrapper = SSDecoderForOV(ss_decoder).eval().float()
    example_input = torch.randn(input_shape, dtype=torch.float32)
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example_input)
    ov.save_model(ov_model, str(output_path))
    print(f"[OV-SAM3D] Saved SS decoder → {output_path}")
    del wrapper
    _cleanup()
    return ov_model


def _unwrap_to_backbone(model):
    """
    Unwrap ShortCut / FlowMatching / ClassifierFreeGuidance wrappers
    to get the actual backbone model (e.g. SparseStructureFlowTdfyWrapper).

    Pipeline model hierarchy:
      ShortCut/FlowMatching
        └─ reverse_fn (ClassifierFreeGuidance*)
            └─ backbone (the actual DiT model)
    """
    # If the model already has t_embedder, it IS the backbone
    if hasattr(model, "t_embedder"):
        return model
    # Try reverse_fn.backbone
    if hasattr(model, "reverse_fn"):
        rf = model.reverse_fn
        if hasattr(rf, "backbone"):
            return rf.backbone
        # reverse_fn might itself be the backbone
        if hasattr(rf, "t_embedder"):
            return rf
    return model


def convert_ss_generator(
    ss_gen_model,
    output_path: Union[str, Path],
    shape_tokens: int = 4096,
    shape_ch: int = 8,
    model_channels: int = 1024,
    cond_channels: int = 1024,
    cond_tokens: int = 1370,
) -> ov.Model:
    """Convert the SS Generator backbone to OpenVINO IR."""
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[OV-SAM3D] Skipping {output_path} — already exists")
        return ov.Core().read_model(str(output_path))

    wrapper = SSGeneratorForOV(ss_gen_model).eval().float()
    wrapper.dtype = torch.float32   # override fp16 dtype after .float()

    # Build example inputs dynamically from the actual latent mapping
    B = 1
    pose_examples = []
    for name in wrapper._pose_names:
        lm = wrapper.latent_mapping[name]
        in_ch = lm.input_layer.in_features
        n_tok = lm.pos_emb.shape[0]
        pose_examples.append(torch.randn(B, n_tok, in_ch))

    example = (
        torch.randn(B, shape_tokens, shape_ch),   # shape_latent
        *pose_examples,                            # pose_0..3 (dynamic dims)
        torch.zeros(B),                            # t
        torch.zeros(B),                            # d
        torch.randn(B, cond_tokens, cond_channels),  # cond
    )
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example)
    ov.save_model(ov_model, str(output_path))
    print(f"[OV-SAM3D] Saved SS generator → {output_path}")
    del wrapper
    _cleanup()
    return ov_model


def convert_slat_generator_core(
    slat_gen_model,
    output_path: Union[str, Path],
    n_voxels: int = 2048,
    model_channels: int = 1024,
    cond_channels: int = 1024,
    cond_tokens: int = 1370,
) -> ov.Model:
    """Convert the SLat Generator core transformer blocks to OpenVINO IR."""
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[OV-SAM3D] Skipping {output_path} — already exists")
        return ov.Core().read_model(str(output_path))

    wrapper = SLatGeneratorCoreForOV(slat_gen_model).eval().float()
    wrapper.dtype = torch.float32   # override fp16 dtype after .float()
    t_emb_dim = 6 * model_channels if wrapper.share_mod else model_channels
    example = (
        torch.randn(n_voxels, model_channels),                  # feats
        torch.randint(0, 32, (n_voxels, 3)),                    # coords_xyz
        torch.randn(1, t_emb_dim),                              # t_emb
        torch.randn(1, cond_tokens, cond_channels),             # cond
    )
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example)
    ov.save_model(ov_model, str(output_path))
    print(f"[OV-SAM3D] Saved SLat generator core → {output_path}")
    del wrapper
    _cleanup()
    return ov_model


def convert_slat_generator_full(
    slat_gen_model,
    output_path: Union[str, Path],
    n_voxels: int = 2048,
    latent_channels: int = 8,
    cond_channels: int = 1024,
    cond_tokens: int = 1370,
) -> ov.Model:
    """
    Convert the **full** SLat Generator backbone (U-Net + core transformer)
    to a single OpenVINO IR.

    This replaces ``convert_slat_generator_core`` and additionally includes
    ``t_embedder``, ``input_layer``, ``input_blocks`` (SparseResBlock3d),
    ``out_blocks``, and ``out_layer`` — eliminating ~46 M PyTorch weight
    parameters.
    """
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[OV-SAM3D] Skipping {output_path} — already exists")
        return ov.Core().read_model(str(output_path))

    wrapper = SLatGeneratorFullForOV(slat_gen_model).eval().float()
    wrapper.dtype = torch.float32   # override fp16 dtype after .float()
    example = (
        torch.randn(n_voxels, latent_channels),                 # feats
        torch.randint(0, 32, (n_voxels, 3)),                    # coords_xyz
        torch.zeros(1),                                          # t
        torch.randn(1, cond_tokens, cond_channels),             # cond
    )
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example)
    ov.save_model(ov_model, str(output_path))
    print(f"[OV-SAM3D] Saved SLat generator full → {output_path}")
    del wrapper
    _cleanup()
    return ov_model


def convert_slat_decoder(
    decoder_model,
    output_path: Union[str, Path],
    n_voxels: int = 2048,
    latent_channels: int = 8,
) -> ov.Model:
    """Convert a SLat decoder (GS / GS-4 / Mesh base) to OpenVINO IR."""
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[OV-SAM3D] Skipping {output_path} — already exists")
        return ov.Core().read_model(str(output_path))

    wrapper = SLatDecoderForOV(decoder_model).eval().float()
    wrapper.dtype = torch.float32   # override fp16 dtype after .float()
    example = (
        torch.randn(n_voxels, latent_channels),   # feats
        torch.randint(0, 64, (n_voxels, 3)),       # coords_xyz
    )
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example)
    ov.save_model(ov_model, str(output_path))
    print(f"[OV-SAM3D] Saved SLat decoder → {output_path}")
    del wrapper
    _cleanup()
    return ov_model


def convert_slat_decoder_mesh_base(
    decoder_model,
    output_path: Union[str, Path],
    n_voxels: int = 2048,
    latent_channels: int = 8,
) -> ov.Model:
    """
    Convert the mesh decoder's **transformer base** (without upsample/out_layer)
    to OpenVINO IR.

    The upsample blocks (SparseSubdivide + SparseConv3d) and FlexiCubes mesh
    extraction remain in PyTorch.
    """
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[OV-SAM3D] Skipping {output_path} — already exists")
        return ov.Core().read_model(str(output_path))

    wrapper = SLatMeshDecoderBaseForOV(decoder_model).eval().float()
    wrapper.dtype = torch.float32
    example = (
        torch.randn(n_voxels, latent_channels),
        torch.randint(0, 64, (n_voxels, 3)),
    )
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example)
    ov.save_model(ov_model, str(output_path))
    print(f"[OV-SAM3D] Saved SLat mesh decoder base → {output_path}")
    del wrapper
    _cleanup()
    return ov_model


def convert_mesh_decoder_upsample(
    decoder_model,
    output_path: Union[str, Path],
    n_voxels: int = 2048,
    model_channels: int = 768,
) -> ov.Model:
    """
    Convert the mesh decoder's **upsample blocks + out_layer** to OV IR.

    Merges ``SparseSubdivideBlock3d × 2`` (5.89 M params) and the
    ``SparseLinear`` out_layer (9 797 params) into a single OV model.

    Input:  ``(N, 768)`` — transformer base output features
    Output: ``(64N, 101)`` — upsampled & projected features
    """
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[OV-SAM3D] Skipping {output_path} — already exists")
        return ov.Core().read_model(str(output_path))

    wrapper = MeshDecoderUpsampleForOV(decoder_model).eval().float()
    example = torch.randn(n_voxels, model_channels)
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example)
    ov.save_model(ov_model, str(output_path))
    print(f"[OV-SAM3D] Saved mesh decoder upsample+out_layer → {output_path}")
    del wrapper
    _cleanup()
    return ov_model


def convert_slat_decoder_mesh_merged(
    decoder_model,
    output_path: Union[str, Path],
    n_voxels: int = 2048,
    latent_channels: int = 8,
) -> ov.Model:
    """
    Convert the mesh decoder's transformer base + upsample + out_layer into
    a **single** OV model (merged from ``SLatMeshDecoderBaseForOV`` +
    ``MeshDecoderUpsampleForOV``).

    Input:  feats ``(N, latent_channels)``, coords_xyz ``(N, 3)``
    Output: ``(64N, out_channels)``  (e.g. 101 for mesh decoder)
    """
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[OV-SAM3D] Skipping {output_path} — already exists")
        return ov.Core().read_model(str(output_path))

    wrapper = SLatMeshDecoderMergedForOV(decoder_model).eval().float()
    example_feats = torch.randn(n_voxels, latent_channels, dtype=torch.float32)
    example_coords = torch.randint(0, 64, (n_voxels, 3), dtype=torch.float32)
    with torch.no_grad():
        ov_model = ov.convert_model(
            wrapper, example_input=(example_feats, example_coords),
        )
    ov.save_model(ov_model, str(output_path))
    print(f"[OV-SAM3D] Saved merged mesh decoder (base+upsample) → {output_path}")
    del wrapper
    _cleanup()
    return ov_model


def convert_moge(
    moge_model,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 3, 700, 700),
) -> ov.Model:
    """Convert the MoGe depth model to OpenVINO IR.

    Default input_shape (1,3,700,700) corresponds to ~2500 DINOv2 tokens
    (50×50 patches of 14×14), matching resolution_level=9.
    """
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[OV-SAM3D] Skipping {output_path} — already exists")
        return ov.Core().read_model(str(output_path))

    wrapper = MoGeForOV(moge_model).eval().float()
    example = torch.randn(input_shape, dtype=torch.float32)
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example)
    ov.save_model(ov_model, str(output_path))
    print(f"[OV-SAM3D] Saved MoGe depth model → {output_path}")

    # Save non-learnable config for runtime
    import json as _json
    config = {
        "traced_h": input_shape[2],
        "traced_w": input_shape[3],
        "remap_output": wrapper.remap_output,
        "mask_threshold": float(getattr(moge_model, "mask_threshold", 0.5)),
        "num_tokens_range": list(getattr(moge_model, "num_tokens_range", [1275, 2551])),
    }
    config_path = output_path.parent / "moge_config.json"
    with open(config_path, "w") as f:
        _json.dump(config, f, indent=2)
    print(f"[OV-SAM3D] Saved MoGe config → {config_path}")

    del wrapper
    _cleanup()
    return ov_model


def convert_point_proj(
    ppe_model,
    output_path: Union[str, Path],
    input_size: int = None,
) -> ov.Model:
    """Convert the PointPatchEmbed outer-path learnable ops to OpenVINO IR.

    Converts ``point_proj`` (nn.Linear) + ``invalid_xyz_token`` application
    into a single OV model.
    """
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[OV-SAM3D] Skipping {output_path} — already exists")
        return ov.Core().read_model(str(output_path))

    if input_size is None:
        input_size = ppe_model.input_size if hasattr(ppe_model, "input_size") else 256

    wrapper = PointProjForOV(ppe_model).eval().float()
    H = W = input_size
    example_xyz = torch.randn(1, H, W, 3, dtype=torch.float32)
    example_mask = torch.ones(1, H, W, dtype=torch.bool)

    with torch.no_grad():
        ov_model = ov.convert_model(
            wrapper, example_input=(example_xyz, example_mask),
        )
    ov.save_model(ov_model, str(output_path))
    print(f"[OV-SAM3D] Saved PointProj → {output_path}")
    del wrapper
    _cleanup()
    return ov_model


def convert_embedder_projection(
    projection_net,
    output_path: Union[str, Path],
    embed_dim: int = None,
    seq_len: int = 1370,
) -> ov.Model:
    """Convert a single EmbedderFuser projection net to OpenVINO IR."""
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[OV-SAM3D] Skipping {output_path} — already exists")
        return ov.Core().read_model(str(output_path))

    # Auto-detect embed_dim from the first LayerNorm or Linear layer
    if embed_dim is None:
        for m in projection_net.modules():
            if hasattr(m, "normalized_shape") and len(m.normalized_shape) > 0:
                embed_dim = m.normalized_shape[0]
                break
            if isinstance(m, nn.Linear):
                embed_dim = m.in_features
                break
        if embed_dim is None:
            embed_dim = 1024  # fallback

    wrapper = EmbedderProjectionForOV(projection_net).eval().float()
    example = torch.randn(1, seq_len, embed_dim, dtype=torch.float32)
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=example)
    ov.save_model(ov_model, str(output_path))
    print(f"[OV-SAM3D] Saved embedder projection → {output_path}")
    del wrapper
    _cleanup()
    return ov_model


def convert_pointpatch_embed_inner(
    ppe_model,
    output_path: Union[str, Path],
    embed_dim: int = None,
    input_size: int = None,
    patch_size: int = None,
) -> ov.Model:
    """Convert the inner attention forward of PointPatchEmbed to OpenVINO IR.

    The outer ``embed_pointmap_windows`` (NaN handling, point_proj, invalid
    token replacement) stays in Python.  This converts only the windowed
    attention + positional embedding portion which contains the bulk of the
    compute and learnable parameters (cls_token, pos_embed_window, pos_embed,
    transformer blocks).

    Also saves the *outer-path* tensor weights (``point_proj``,
    ``invalid_xyz_token``, ``dropped_xyz_token``) to a sidecar JSON file
    so the OV pipeline can run without the original PyTorch checkpoint.
    """
    import json

    output_path = Path(output_path)
    if output_path.exists():
        print(f"[OV-SAM3D] Skipping {output_path} — already exists")
        return ov.Core().read_model(str(output_path))

    # Auto-detect dimensions from the module
    if embed_dim is None:
        embed_dim = ppe_model.embed_dim
    if input_size is None:
        input_size = ppe_model.input_size if hasattr(ppe_model, "input_size") else 256
    if patch_size is None:
        patch_size = ppe_model.patch_size

    wrapper = PointPatchEmbedInnerForOV(ppe_model).eval().float()

    # Fixed-size example inputs (pipeline always resizes to input_size)
    H = W = input_size
    example_x = torch.randn(1, H, W, embed_dim, dtype=torch.float32)
    example_nh = torch.tensor(H, dtype=torch.int64)
    example_nw = torch.tensor(W, dtype=torch.int64)

    with torch.no_grad():
        ov_model = ov.convert_model(
            wrapper, example_input=(example_x, example_nh, example_nw),
        )
    ov.save_model(ov_model, str(output_path))
    print(f"[OV-SAM3D] Saved PointPatchEmbed inner → {output_path}")

    # ── Save outer-path weights to sidecar JSON ──────────────────────
    config_path = output_path.with_name("pointpatch_config.json")
    outer_weights: Dict[str, Any] = {}
    # point_proj (nn.Linear)
    if hasattr(ppe_model, "point_proj"):
        outer_weights["point_proj_weight"] = ppe_model.point_proj.weight.detach().cpu().float().tolist()
        if ppe_model.point_proj.bias is not None:
            outer_weights["point_proj_bias"] = ppe_model.point_proj.bias.detach().cpu().float().tolist()
    # invalid_xyz_token
    if hasattr(ppe_model, "invalid_xyz_token"):
        outer_weights["invalid_xyz_token"] = ppe_model.invalid_xyz_token.detach().cpu().float().tolist()
    # dropped_xyz_token (optional)
    if hasattr(ppe_model, "dropped_xyz_token"):
        outer_weights["dropped_xyz_token"] = ppe_model.dropped_xyz_token.detach().cpu().float().tolist()
    # remap_type for PointRemapper (stateless but need the config)
    if hasattr(ppe_model, "point_remapper"):
        outer_weights["remap_type"] = ppe_model.point_remapper.remap_type
    # input_size, patch_size, embed_dim for reconstruction
    outer_weights["input_size"] = input_size
    outer_weights["patch_size"] = patch_size
    outer_weights["embed_dim"] = embed_dim

    with open(config_path, "w") as f:
        json.dump(outer_weights, f)
    print(f"[OV-SAM3D] Saved PointPatchEmbed outer weights → {config_path}")

    del wrapper
    _cleanup()
    return ov_model


def convert_all_models(
    pipeline,
    output_dir: Union[str, Path],
    device: str = "CPU",
) -> Dict[str, Any]:
    """
    Convert all convertible sub-models from the pipeline to OpenVINO IR,
    save them to ``output_dir``, and return a dict of compiled models.

    Parameters
    ----------
    pipeline : InferencePipelinePointMap
        The fully-initialised (CPU) pipeline.
    output_dir : str | Path
        Directory to write ``.xml`` / ``.bin`` files.
    device : str
        OpenVINO device (``CPU``, ``GPU``, …).

    Returns
    -------
    dict
        Keys like ``ss_dino_image_ov``, ``ss_dino_mask_ov``, ``ss_decoder_ov``, etc.
        Values are compiled ``ov.CompiledModel`` objects.
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    compiled = {}

    # ── Merged DINOv2 backbone (shared across all 4 embedders) ─────────
    # All four DINOv2 embedders (SS-image, SS-mask, SLat-image, SLat-mask)
    # share identical ViT-L/14 weights.  Convert once → ~3.6 GB savings.
    ss_embedder = pipeline.condition_embedders["ss_condition_embedder"]
    ss_dino_image = ss_embedder.embedder_list[0][0]   # any of the 4 will do
    print("[OV-SAM3D] Converting merged DINOv2 backbone (1 model for all 4 embedders) …")
    ov_m = convert_dino_backbone(ss_dino_image, output_dir / "dino_backbone.xml")
    compiled["dino_backbone_ov"] = core.compile_model(ov_m, device)

    # ── PointPatchEmbed inner attention (SS embedder only) ─────────────
    if len(ss_embedder.embedder_list) > 2:
        ppe = ss_embedder.embedder_list[2][0]  # the PointPatchEmbed module
        print("[OV-SAM3D] Converting PointPatchEmbed inner attention …")
        ov_m = convert_pointpatch_embed_inner(
            ppe, output_dir / "pointpatch_embed_inner.xml",
        )
        compiled["pointpatch_embed_inner_ov"] = core.compile_model(ov_m, device)

    # ── SS decoder ─────────────────────────────────────────────────────
    print("[OV-SAM3D] Converting SS decoder …")
    ss_decoder = pipeline.models["ss_decoder"]
    ov_m = convert_ss_decoder(ss_decoder, output_dir / "ss_decoder.xml")
    compiled["ss_decoder_ov"] = core.compile_model(ov_m, device)

    # ── SS generator backbone ──────────────────────────────────────────
    print("[OV-SAM3D] Converting SS generator backbone …")
    ss_gen = pipeline.models["ss_generator"]
    # Unwrap ShortCut / ClassifierFreeGuidance → backbone
    ss_gen_backbone = _unwrap_to_backbone(ss_gen)
    ov_m = convert_ss_generator(ss_gen_backbone, output_dir / "ss_generator.xml")
    compiled["ss_generator_ov"] = core.compile_model(ov_m, device)

    # ── SLat generator full (U-Net + core transformer in one model) ──
    print("[OV-SAM3D] Converting SLat generator full backbone …")
    slat_gen = pipeline.models["slat_generator"]
    # Unwrap FlowMatching / ClassifierFreeGuidance → backbone
    slat_gen_backbone = _unwrap_to_backbone(slat_gen)
    ov_m = convert_slat_generator_full(slat_gen_backbone, output_dir / "slat_generator_full.xml")
    compiled["slat_generator_full_ov"] = core.compile_model(ov_m, device)

    # ── SLat decoders (gaussian, gaussian-4) ──────────────────────────
    slat_decoder_keys = [
        k for k in pipeline.models
        if k.startswith("slat_decoder_") and k not in ("slat_decoder_mesh",)
    ]
    for dkey in slat_decoder_keys:
        safe_name = dkey.replace("/", "_")
        print(f"[OV-SAM3D] Converting {dkey} …")
        ov_m = convert_slat_decoder(
            pipeline.models[dkey], output_dir / f"{safe_name}.xml",
        )
        compiled[f"{safe_name}_ov"] = core.compile_model(ov_m, device)

    # ── SLat mesh decoder (transformer base only) ──────────────────
    if "slat_decoder_mesh" in pipeline.models:
        # ── Try merged mesh decoder first (base + upsample in one model) ──
        print("[OV-SAM3D] Converting slat_decoder_mesh (merged: base + upsample) …")
        try:
            ov_m = convert_slat_decoder_mesh_merged(
                pipeline.models["slat_decoder_mesh"],
                output_dir / "slat_decoder_mesh_merged.xml",
            )
            compiled["slat_decoder_mesh_merged_ov"] = core.compile_model(ov_m, device)
        except Exception as e:
            print(f"[OV-SAM3D] Merged mesh decoder failed ({e}), falling back to separate models")
            # Fallback: separate base + upsample models
            print("[OV-SAM3D] Converting slat_decoder_mesh (transformer base) …")
            ov_m = convert_slat_decoder_mesh_base(
                pipeline.models["slat_decoder_mesh"],
                output_dir / "slat_decoder_mesh.xml",
            )
            compiled["slat_decoder_mesh_ov"] = core.compile_model(ov_m, device)

            print("[OV-SAM3D] Converting mesh decoder upsample + out_layer …")
            ov_m = convert_mesh_decoder_upsample(
                pipeline.models["slat_decoder_mesh"],
                output_dir / "mesh_decoder_upsample.xml",
            )
            compiled["mesh_decoder_upsample_ov"] = core.compile_model(ov_m, device)

    # ── MoGe depth model ──────────────────────────────────────────────
    # Convert MoGe to OV.  The mock from patch_cuda_for_cpu() has no real
    # weights, so we load the real model first.
    print("[OV-SAM3D] Converting MoGe depth model …")
    try:
        real_moge = load_real_moge_cpu()
        ov_m = convert_moge(real_moge, output_dir / "moge_depth.xml")
        compiled["moge_depth_ov"] = core.compile_model(ov_m, device)
        del real_moge
        _cleanup()
    except Exception as e:
        print(f"[OV-SAM3D] WARNING: MoGe OV conversion failed ({e}), "
              "will fall back to PyTorch on CPU")

    # ── PointPatchEmbed outer-path point_proj (nn.Linear) ─────────────
    if len(ss_embedder.embedder_list) > 2:
        ppe = ss_embedder.embedder_list[2][0]
        if hasattr(ppe, "point_proj"):
            print("[OV-SAM3D] Converting PointPatchEmbed point_proj …")
            ov_m = convert_point_proj(ppe, output_dir / "point_proj.xml")
            compiled["point_proj_ov"] = core.compile_model(ov_m, device)

    # ── EmbedderFuser idx_emb (learned positional embeddings) ─────────
    # These nn.Parameters live in EmbedderFuser and are added to condition
    # tokens during forward().  They are NOT part of any OV model, so we
    # save them to a sidecar JSON so the OV pipeline can load them without
    # needing the original PyTorch checkpoint.
    embedder_fuser_config: Dict[str, Any] = {}
    for stage_name in ["ss_condition_embedder", "slat_condition_embedder"]:
        if stage_name in pipeline.condition_embedders:
            emb = pipeline.condition_embedders[stage_name]
            if hasattr(emb, "idx_emb") and emb.idx_emb is not None:
                embedder_fuser_config[f"{stage_name}_idx_emb"] = (
                    emb.idx_emb.detach().cpu().float().tolist()
                )
                embedder_fuser_config[f"{stage_name}_use_pos_embedding"] = (
                    emb.use_pos_embedding
                )
    if embedder_fuser_config:
        _ef_cfg_path = output_dir / "embedder_fuser_config.json"
        with open(_ef_cfg_path, "w") as _f:
            json.dump(embedder_fuser_config, _f)
        print(f"[OV-SAM3D] Saved EmbedderFuser idx_emb → {_ef_cfg_path}")

    # ── EmbedderFuser projection nets ──────────────────────────────────
    for stage_name in ["ss_condition_embedder", "slat_condition_embedder"]:
        if stage_name in pipeline.condition_embedders:
            emb = pipeline.condition_embedders[stage_name]
            if hasattr(emb, "projection_nets"):
                for i, proj_net in enumerate(emb.projection_nets):
                    proj_key = f"{stage_name}_proj_{i}"
                    print(f"[OV-SAM3D] Converting {proj_key} …")
                    ov_m = convert_embedder_projection(
                        proj_net, output_dir / f"{proj_key}.xml",
                    )
                    compiled[f"{proj_key}_ov"] = core.compile_model(ov_m, device)

    _cleanup()

    # Save lightweight sidecar data for standalone OV pipeline creation
    _save_pipeline_sidecar(pipeline, output_dir)

    print(f"[OV-SAM3D] All models saved to {output_dir}")
    return compiled


def _save_pipeline_sidecar(pipeline, output_dir: Path):
    """Save lightweight pipeline metadata needed to create OV pipeline.

    All learnable weights are in OV models.  This JSON sidecar stores only:
      - Generator backbone metadata (force_zeros_cond, latent_share_transformer)
      - Latent mapping shapes (pos_emb sizes for noise generation)
    """
    import json as _json

    output_dir = Path(output_dir)
    sidecar: Dict[str, Any] = {}

    # ── Generator backbone metadata + latent_mapping shapes ──
    for gen_name in ["ss_generator", "slat_generator"]:
        gen = _moduledict_get(pipeline.models, gen_name)
        if gen is None:
            continue
        backbone = _unwrap_to_backbone(gen)
        if backbone is None:
            continue

        prefix = gen_name

        # Save metadata (no learnable weights — just config)
        meta: Dict[str, Any] = {}
        if hasattr(backbone, "force_zeros_cond"):
            meta["force_zeros_cond"] = bool(backbone.force_zeros_cond)
        if hasattr(backbone, "latent_share_transformer"):
            for k, v in backbone.latent_share_transformer.items():
                meta["latent_share_transformer"] = {
                    str(k): list(v) if not isinstance(v, list) else v
                }
                break
        sidecar[f"{prefix}_meta"] = meta

        # Save latent_mapping shapes only (weights are in OV models)
        if hasattr(backbone, "latent_mapping"):
            lm_shapes: Dict[str, Any] = {}
            for entry_name in backbone.latent_mapping:
                entry = backbone.latent_mapping[entry_name]
                entry_info: Dict[str, Any] = {}
                if hasattr(entry, "pos_emb"):
                    entry_info["pos_emb_shape"] = list(entry.pos_emb.shape)
                if hasattr(entry, "input_layer"):
                    entry_info["input_in_features"] = entry.input_layer.in_features
                    entry_info["input_out_features"] = entry.input_layer.out_features
                if hasattr(entry, "output_layer"):
                    entry_info["output_in_features"] = entry.output_layer.in_features
                    entry_info["output_out_features"] = entry.output_layer.out_features
                lm_shapes[entry_name] = entry_info
            sidecar[f"{prefix}_latent_mapping_shapes"] = lm_shapes

    sidecar_path = output_dir / "pipeline_sidecar.json"
    with open(sidecar_path, "w") as f:
        _json.dump(sidecar, f, indent=2)
    print(f"[OV-SAM3D] Saved pipeline sidecar metadata to {sidecar_path}")


# ============================================================================
# 4.  OV INFERENCE WRAPPERS — drop-in replacements that call compiled OV models
# ============================================================================

class OVDinoEmbedder:
    """
    Replaces a ``Dino`` nn.Module with an OpenVINO compiled-model wrapper.

    Works with the **merged** DINOv2 backbone that has two outputs:
        Output 0 — postnorm (for SS stage, prenorm=False)
        Output 1 — prenorm  (for SLat stage, prenorm=True)

    For mask inputs, repeats 1-channel → 3-channel before calling the model.

    Preserves the same ``__call__(x) → tokens`` API expected by ``EmbedderFuser``.
    """

    def __init__(
        self,
        compiled_model: ov.CompiledModel,
        embed_dim: int,
        is_mask: bool = False,
        prenorm: bool = False,
    ):
        self.compiled_model = compiled_model
        self.embed_dim = embed_dim
        self.is_mask = is_mask
        self.prenorm = prenorm  # True → SLat stage (output 1), False → SS stage (output 0)
        self._output_index = 1 if prenorm else 0
        self._infer_request = compiled_model.create_infer_request()

    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        orig_dtype = x.dtype
        orig_device = x.device
        # For mask inputs the merged backbone expects 3-channel input,
        # so repeat 1ch → 3ch here rather than inside the OV graph.
        if self.is_mask and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x_np = x.float().detach().cpu().numpy()
        self._infer_request.infer(x_np)
        out_np = self._infer_request.get_output_tensor(self._output_index).data
        return torch.from_numpy(out_np.copy()).to(dtype=orig_dtype, device=orig_device)

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self


class OVSSDecoder(nn.Module):
    """
    Replaces ``SparseStructureDecoder`` with an OpenVINO compiled model.

    Preserves the ``forward(x) → occupancy`` API.
    """

    def __init__(self, compiled_model: ov.CompiledModel):
        super().__init__()
        self.compiled_model = compiled_model
        self._infer_request = compiled_model.create_infer_request()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.float().detach().cpu().numpy()
        self._infer_request.infer(x_np)
        out_np = self._infer_request.get_output_tensor(0).data
        return torch.from_numpy(out_np.copy()).to(dtype=x.dtype, device=x.device)


class OVSSGenerator:
    """
    Replaces the ``SparseStructureFlowModel`` backbone with an OV compiled model.

    Wraps the conversion wrapper's interface:
    ``(shape_latent, pose_trans, pose_scale, pose_rot, pose_ts, t, d, cond)
    → (shape_out, trans_out, scale_out, rot_out, ts_out)``
    """

    def __init__(self, compiled_model: ov.CompiledModel):
        self.compiled_model = compiled_model
        self._infer_request = compiled_model.create_infer_request()

    def __call__(
        self,
        shape_latent: torch.Tensor,
        pose_trans: torch.Tensor,
        pose_scale: torch.Tensor,
        pose_rot: torch.Tensor,
        pose_ts: torch.Tensor,
        t: torch.Tensor,
        d: torch.Tensor,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        inputs = [
            a.float().detach().cpu().numpy()
            for a in (shape_latent, pose_trans, pose_scale, pose_rot, pose_ts, t, d, cond)
        ]
        self._infer_request.infer(inputs)
        outputs = []
        n_outputs = len(self.compiled_model.outputs)
        for i in range(n_outputs):
            out_np = self._infer_request.get_output_tensor(i).data
            outputs.append(torch.from_numpy(out_np.copy()))
        return tuple(outputs)


class OVSLatGeneratorCore:
    """
    Replaces the SLat generator's core transformer blocks with an OV model.

    ``(feats, coords_xyz, t_emb, cond) → out_feats``
    """

    def __init__(self, compiled_model: ov.CompiledModel):
        self.compiled_model = compiled_model
        self._infer_request = compiled_model.create_infer_request()

    def __call__(
        self,
        feats: torch.Tensor,
        coords_xyz: torch.Tensor,
        t_emb: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        feats_np = feats.float().detach().cpu().numpy()
        coords_np = coords_xyz.float().detach().cpu().numpy()
        t_emb_np = t_emb.float().detach().cpu().numpy()
        cond_np = cond.float().detach().cpu().numpy()
        self._infer_request.infer([feats_np, coords_np, t_emb_np, cond_np])
        out_np = self._infer_request.get_output_tensor(0).data
        return torch.from_numpy(out_np.copy())


class OVSLatGeneratorFull:
    """
    Replaces the **full** SLat generator backbone with a single OV model.

    ``(feats, coords_xyz, t, cond) → out_feats``

    The OV model includes t_embedder, input_layer, input_blocks (dense ResBlocks),
    24 core transformer blocks, output_blocks, and out_layer — all weights in OV.
    """

    def __init__(self, compiled_model: ov.CompiledModel):
        self.compiled_model = compiled_model
        self._infer_request = compiled_model.create_infer_request()

    def __call__(
        self,
        feats: torch.Tensor,
        coords_xyz: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        feats_np = feats.float().detach().cpu().numpy()
        coords_np = coords_xyz.float().detach().cpu().numpy()
        t_np = t.float().detach().cpu().numpy()
        cond_np = cond.float().detach().cpu().numpy()
        self._infer_request.infer([feats_np, coords_np, t_np, cond_np])
        out_np = self._infer_request.get_output_tensor(0).data
        return torch.from_numpy(out_np.copy())


class OVSLatDecoder:
    """
    Replaces the SLat decoder's transformer base with an OV model.

    ``(feats, coords_xyz) → out_feats``
    """

    def __init__(self, compiled_model: ov.CompiledModel):
        self.compiled_model = compiled_model
        self._infer_request = compiled_model.create_infer_request()

    def __call__(
        self,
        feats: torch.Tensor,
        coords_xyz: torch.Tensor,
    ) -> torch.Tensor:
        feats_np = feats.float().detach().cpu().numpy()
        coords_np = coords_xyz.float().detach().cpu().numpy()
        self._infer_request.infer([feats_np, coords_np])
        out_np = self._infer_request.get_output_tensor(0).data
        return torch.from_numpy(out_np.copy())


class OVMeshDecoderUpsample:
    """
    OV wrapper for the merged mesh-decoder upsample blocks + out_layer.

    ``(feats) → out_feats``

    Input:  ``(N, 768)``
    Output: ``(64N, 101)``
    """

    def __init__(self, compiled_model: ov.CompiledModel):
        self.compiled_model = compiled_model
        self._infer_request = compiled_model.create_infer_request()

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        feats_np = feats.float().detach().cpu().numpy()
        self._infer_request.infer(feats_np)
        out_np = self._infer_request.get_output_tensor(0).data
        return torch.from_numpy(out_np.copy())


class OVMeshDecoderMerged:
    """
    OV wrapper for the **merged** mesh decoder (base transformer + upsample
    + out_layer in a single OV model).

    Input:  feats ``(N, latent_ch)``, coords_xyz ``(N, 3)``
    Output: ``(64N, out_ch)``
    """

    def __init__(self, compiled_model: ov.CompiledModel):
        self.compiled_model = compiled_model
        self._infer_request = compiled_model.create_infer_request()

    def __call__(
        self, feats: torch.Tensor, coords_xyz: torch.Tensor,
    ) -> torch.Tensor:
        feats_np = feats.float().detach().cpu().numpy()
        coords_np = coords_xyz.float().detach().cpu().numpy()
        self._infer_request.infer([feats_np, coords_np])
        out_np = self._infer_request.get_output_tensor(0).data
        return torch.from_numpy(out_np.copy())


class OVPointPatchEmbedInner:
    """
    Replaces the inner forward of ``PointPatchEmbed`` (windowed attention +
    positional embedding) with an OV compiled model.

    Input:  x (B, H, W, embed_dim), n_h (scalar), n_w (scalar)
    Output: (B, n_windows, embed_dim)
    """

    def __init__(self, compiled_model: ov.CompiledModel):
        self.compiled_model = compiled_model
        self._infer_request = compiled_model.create_infer_request()

    def __call__(
        self, x: torch.Tensor, n_h: torch.Tensor, n_w: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_np = x.float().detach().cpu().numpy()
        nh_np = np.array(int(n_h.item()), dtype=np.int64)
        nw_np = np.array(int(n_w.item()), dtype=np.int64)
        self._infer_request.infer([x_np, nh_np, nw_np])
        out_np = self._infer_request.get_output_tensor(0).data
        return torch.from_numpy(out_np.copy()).to(dtype=orig_dtype)


class OVMoGe:
    """
    Replaces the MoGe depth model with an OV model.

    Replicates the full ``MoGeModel.infer()`` pipeline:
      1. Resize input image to num_tokens-determined resolution
      2. Call OV model (normalization + backbone + head + _remap_points)
      3. Resize output back to original resolution
      4. Apply recover_focal_shift + z-shift + mask

    Returns the same dict as ``MoGe(DepthModel).__call__``:
      ``{"pointmaps": (H, W, 3), "intrinsics": (3, 3), "mask": (H, W)}``
    """

    def __init__(self, compiled_model: ov.CompiledModel, config: dict = None):
        self.compiled_model = compiled_model
        self._infer_request = compiled_model.create_infer_request()
        if config is None:
            config = {}
        self.traced_h = config.get("traced_h", 700)
        self.traced_w = config.get("traced_w", 700)
        self.mask_threshold = config.get("mask_threshold", 0.5)
        self.num_tokens_range = config.get("num_tokens_range", [1275, 2551])

    def __call__(
        self,
        image: torch.Tensor,
        resolution_level: int = 9,
        force_projection: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        image : (B, 3, H, W) or (3, H, W)  float32 in [0, 1]
        resolution_level : int  [0-9], controls DINOv2 token count.
        force_projection : bool  if True, recompute pointmap from depth.

        Returns
        -------
        dict with keys: pointmaps, intrinsics, depth, mask, mask_prob
        """
        if image.dim() == 3:
            omit_batch = True
            image = image.unsqueeze(0)
        else:
            omit_batch = False

        _, _, orig_h, orig_w = image.shape

        # Resize to traced resolution (matches num_tokens at trace time)
        if orig_h != self.traced_h or orig_w != self.traced_w:
            image_resized = F.interpolate(
                image.float(), (self.traced_h, self.traced_w),
                mode="bilinear", align_corners=False,
            )
        else:
            image_resized = image.float()

        # Run OV model → (points, mask)
        image_np = image_resized.detach().cpu().numpy()
        self._infer_request.infer(image_np)
        points_np = self._infer_request.get_output_tensor(0).data
        mask_np = self._infer_request.get_output_tensor(1).data
        points = torch.from_numpy(points_np.copy())  # (B, H_t, W_t, 3)
        mask = torch.from_numpy(mask_np.copy())       # (B, H_t, W_t)

        # Resize back to original spatial dims
        if orig_h != self.traced_h or orig_w != self.traced_w:
            points = points.permute(0, 3, 1, 2)  # (B, 3, H_t, W_t)
            points = F.interpolate(
                points, (orig_h, orig_w),
                mode="bilinear", align_corners=False,
            )
            points = points.permute(0, 2, 3, 1)  # (B, H, W, 3)
            mask = F.interpolate(
                mask.unsqueeze(1), (orig_h, orig_w),
                mode="bilinear", align_corners=False,
            ).squeeze(1)

        mask_binary = mask > self.mask_threshold

        # Post-processing: recover_focal_shift + depth (matches MoGeModel.infer)
        try:
            from moge.utils.geometry_torch import recover_focal_shift
            import utils3d.torch as u3d

            aspect_ratio = orig_w / orig_h
            focal, shift = recover_focal_shift(points, mask_binary)
            fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
            fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
            intrinsics = u3d.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
            depth = points[..., 2] + shift[..., None, None]

            if force_projection:
                points = u3d.depth_to_points(depth, intrinsics=intrinsics)
            else:
                points = points + torch.stack(
                    [torch.zeros_like(shift), torch.zeros_like(shift), shift],
                    dim=-1,
                )[..., None, None, :]

            # Apply mask
            points = torch.where(mask_binary[..., None], points, torch.tensor(float("inf")))
            depth = torch.where(mask_binary, depth, torch.tensor(float("inf")))
        except Exception as _e:
            print(f"[OV-SAM3D] WARNING: MoGe post-processing failed ({_e}), "
                  "returning raw points")
            focal_fallback = float(max(orig_h, orig_w))
            intrinsics = torch.tensor([
                [focal_fallback, 0.0, orig_w / 2.0],
                [0.0, focal_fallback, orig_h / 2.0],
                [0.0, 0.0, 1.0],
            ], dtype=torch.float32).unsqueeze(0)
            depth = points[..., 2]

        if omit_batch:
            points = points.squeeze(0)
            intrinsics = intrinsics.squeeze(0)
            depth = depth.squeeze(0)
            mask_binary = mask_binary.squeeze(0)
            mask = mask.squeeze(0)

        return {
            "pointmaps": points,
            "points": points,
            "intrinsics": intrinsics,
            "depth": depth,
            "mask": mask_binary,
            "mask_prob": torch.sigmoid(mask),
        }


class OVPointProj:
    """
    Replaces the PointPatchEmbed outer-path learnable ops with an OV model.

    ``(xyz_remapped, valid_mask) → projected embeddings``
    """

    def __init__(self, compiled_model):
        self.compiled_model = compiled_model
        self._infer_request = compiled_model.create_infer_request()

    def __call__(self, xyz_remapped: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        xyz_np = xyz_remapped.float().detach().cpu().numpy()
        mask_np = valid_mask.detach().cpu().numpy()
        self._infer_request.infer([xyz_np, mask_np])
        out_np = self._infer_request.get_output_tensor(0).data
        return torch.from_numpy(out_np.copy())


class OVEmbedderProjection(nn.Module):
    """
    Replaces an ``EmbedderFuser`` projection net (LayerNorm → FeedForward)
    with an OV compiled model.

    ``(B, L, embed_dim) → (B, L, output_dim)``
    """

    def __init__(self, compiled_model: ov.CompiledModel):
        super().__init__()
        self.compiled_model = compiled_model
        self._infer_request = compiled_model.create_infer_request()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.float().detach().cpu().numpy()
        self._infer_request.infer(x_np)
        out_np = self._infer_request.get_output_tensor(0).data
        return torch.from_numpy(out_np.copy()).to(dtype=x.dtype, device=x.device)


# ============================================================================
# 5.  OV PIPELINE — modified InferencePipelinePointMap that uses OV models
# ============================================================================

def _get_inference_pipeline_pointmap_class():
    """Lazy import to avoid import-order issues with patch_cuda_for_cpu()."""
    from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
    return InferencePipelinePointMap


# The real class (inheriting from InferencePipelinePointMap) is built
# lazily by ``_ensure_ov_pipeline_class_ready()`` — the first call to
# ``OVInferencePipelinePointMap(...)`` triggers it.  At module-import
# time we only define a thin wrapper that defers class creation until
# ``patch_cuda_for_cpu()`` has installed the required mocks.

class _OVPipelineMixin:
    """Holds ALL OVInferencePipelinePointMap methods.

    Defined at module level (no sam3d_objects dependency).
    ``_ensure_ov_pipeline_class()`` dynamically sets the real base
    class (InferencePipelinePointMap) onto this mixin via ``__bases__``
    manipulation — see below.
    """

    def __init__(
        self,
        compiled_models: Dict[str, Any],
        ov_model_dir: Union[str, Path] = "./ov_models",
        *,
        config_path: Union[str, Path, None] = None,
    ):
        """
        Parameters
        ----------
        compiled_models : dict
            Dict returned by ``convert_all_models`` or ``load_compiled_models``.
        ov_model_dir : str | Path
            Directory containing OV IR files and sidecar configs.
        config_path : str | Path
            Path to ``pipeline.yaml``.  The pipeline skeleton is recreated
            from this config **without loading heavy checkpoint weights**,
            then small sidecar weights are restored.
        """
        if config_path is None:
            raise ValueError(
                "config_path is required. Pass the path to pipeline.yaml."
            )
        self._init_lightweight(config_path, ov_model_dir)
        self._compiled = compiled_models
        self._ov_model_dir = Path(ov_model_dir)
        self._patch_embedders()
        self._patch_ss_decoder()
        self._patch_ss_generator()
        self._patch_slat_generator()
        self._patch_slat_decoders()
        self._patch_moge()
        self._patch_autocast()
        # Ensure all modules are in eval mode — in training mode, PyTorch
        # CFG wrappers skip guidance blending and only call backbone once,
        # which drastically degrades quality.
        self.models.eval()
        self._free_replaced_parameters()

    # ------------------------------------------------------------------
    #  Lightweight initialization from config (no heavy weights)
    # ------------------------------------------------------------------
    def _init_lightweight(self, config_path, ov_model_dir):
        """Recreate pipeline skeleton from config without loading checkpoint weights."""
        import warnings
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        import safetensors.torch as _sft

        config_path = Path(config_path)
        ov_model_dir = Path(ov_model_dir)

        config = OmegaConf.load(str(config_path))
        config = patch_pipeline_config(config)
        config.workspace_dir = str(config_path.parent)

        # Patch ONLY the SAM3D model loading to skip weights — DINOv2 hub
        # loading (torch.hub.load) must still work because it creates the
        # ViT architecture.  We do NOT replace torch.load globally.
        _orig_load_file = _sft.load_file
        _sft.load_file = lambda *_a, **_kw: {}

        # Patch load_model_from_checkpoint to return model with empty state dict
        from sam3d_objects.model import io as _io_mod
        from sam3d_objects.pipeline import inference_pipeline as _ip_mod2
        _orig_load_ckpt = _io_mod.load_model_from_checkpoint

        def _noop_load_ckpt(model, ckpt_path, **kwargs):
            """Skip checkpoint loading — return model as-is."""
            return model

        _io_mod.load_model_from_checkpoint = _noop_load_ckpt
        _ip_mod2.load_model_from_checkpoint = _noop_load_ckpt

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skeleton = instantiate(config)
        finally:
            _sft.load_file = _orig_load_file
            _io_mod.load_model_from_checkpoint = _orig_load_ckpt
            _ip_mod2.load_model_from_checkpoint = _orig_load_ckpt

        # Transfer skeleton state to self
        self.__dict__.update(skeleton.__dict__)
        del skeleton

        # Restore small needed weights from sidecar
        self._restore_sidecar_weights(ov_model_dir)
        print("[OV-SAM3D] Lightweight pipeline created from config (no heavy weights loaded)")

    def _restore_sidecar_weights(self, ov_model_dir):
        """Load pipeline metadata from JSON sidecar (no .pt files).

        All learnable weights are in OV models.  This only restores:
          - Generator backbone metadata (force_zeros_cond, latent_share_transformer)
          - Latent mapping shapes (pos_emb sizes for noise generation)
        """
        import json as _json

        sidecar_path = Path(ov_model_dir) / "pipeline_sidecar.json"
        if not sidecar_path.exists():
            # Try legacy .pt file for backward compatibility
            legacy_path = Path(ov_model_dir) / "pipeline_sidecar.pt"
            if legacy_path.exists():
                print(f"[OV-SAM3D] WARNING: Found legacy {legacy_path}, "
                      "please re-run convert_all_models to generate JSON sidecar")
                self._restore_sidecar_weights_legacy(legacy_path)
                return
            print(f"[OV-SAM3D] WARNING: {sidecar_path} not found, "
                  "skipping sidecar restoration")
            return

        with open(sidecar_path, "r") as f:
            sidecar = _json.load(f)

        # Restore generator backbone metadata
        for gen_name in ["ss_generator", "slat_generator"]:
            meta_key = f"{gen_name}_meta"
            if meta_key not in sidecar:
                continue
            meta = sidecar[meta_key]
            gen = _moduledict_get(self.models, gen_name)
            if gen is None:
                continue
            backbone = _unwrap_to_backbone(gen)
            if backbone is None:
                continue
            if "force_zeros_cond" in meta:
                backbone.force_zeros_cond = meta["force_zeros_cond"]
            if "latent_share_transformer" in meta:
                # Rebuild the dict structure
                for merged_key, names in meta["latent_share_transformer"].items():
                    backbone.latent_share_transformer = {merged_key: names}
                    break

        # Restore latent_mapping shapes (create dummy tensors with correct sizes)
        for gen_name in ["ss_generator", "slat_generator"]:
            shapes_key = f"{gen_name}_latent_mapping_shapes"
            if shapes_key not in sidecar:
                continue
            gen = _moduledict_get(self.models, gen_name)
            if gen is None:
                continue
            backbone = _unwrap_to_backbone(gen)
            if backbone is None or not hasattr(backbone, "latent_mapping"):
                continue
            lm_shapes = sidecar[shapes_key]
            for entry_name, entry_info in lm_shapes.items():
                if entry_name not in backbone.latent_mapping:
                    continue
                entry = backbone.latent_mapping[entry_name]
                # Fix pos_emb shape (skeleton may have empty/wrong shape)
                if "pos_emb_shape" in entry_info and hasattr(entry, "pos_emb"):
                    target_shape = entry_info["pos_emb_shape"]
                    if list(entry.pos_emb.shape) != target_shape:
                        dummy = torch.zeros(target_shape, dtype=entry.pos_emb.dtype)
                        if isinstance(entry.pos_emb, nn.Parameter):
                            entry.pos_emb = nn.Parameter(dummy, requires_grad=False)
                        else:
                            entry.pos_emb = dummy

        print(f"[OV-SAM3D] Restored pipeline metadata from {sidecar_path}")

    def _restore_sidecar_weights_legacy(self, sidecar_path):
        """Backward-compatible loader for legacy .pt sidecar files."""
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)

        for gen_name in ["ss_generator", "slat_generator"]:
            lm_key = f"{gen_name}_latent_mapping"
            if lm_key not in sidecar:
                continue
            gen = _moduledict_get(self.models, gen_name)
            if gen is None:
                continue
            backbone = _unwrap_to_backbone(gen)
            if backbone is None or not hasattr(backbone, "latent_mapping"):
                continue
            saved_lm = sidecar[lm_key]
            for param_name, saved_tensor in saved_lm.items():
                parts = param_name.split(".")
                mod = backbone.latent_mapping
                for part in parts[:-1]:
                    mod = getattr(mod, part)
                attr = parts[-1]
                old = getattr(mod, attr, None)
                if old is not None and isinstance(old, nn.Parameter):
                    setattr(mod, attr, nn.Parameter(saved_tensor, requires_grad=False))
                elif old is not None and isinstance(old, torch.Tensor):
                    setattr(mod, attr, saved_tensor)

        print(f"[OV-SAM3D] Restored legacy sidecar weights from {sidecar_path}")

    def _free_replaced_parameters(self):
        """Free PyTorch parameters that have been replaced by OV compiled models."""
        import gc

        freed = 0
        # Free DINOv2 backbone weights inside condition embedders
        # (replaced by OVDinoEmbedder — the original nn.Module is gone)
        # Nothing to do here — embedder_list entries are replaced objects.

        # Free generator backbone core transformer weights
        # (replaced by OV, but condition_embedder is still needed)
        for gen_name in ["ss_generator", "slat_generator"]:
            gen = _moduledict_get(self.models, gen_name)
            if gen is None:
                continue
            backbone = _unwrap_to_backbone(gen)
            if backbone is None:
                continue
            kept_prefixes = {"condition_embedder", "latent_mapping"}
            # For core-only SLat mode, also keep I/O blocks
            if gen_name == "slat_generator" and not hasattr(self, "_ov_slat_generator_full"):
                kept_prefixes.update({
                    "input_layer", "t_embedder", "d_embedder",
                    "adaLN_modulation", "input_blocks", "out_blocks",
                    "out_layer", "pos_embedder",
                })
            for name, param in list(backbone.named_parameters()):
                top = name.split(".")[0]
                if top not in kept_prefixes:
                    n = param.numel() * param.element_size()
                    param.data = torch.empty(0, dtype=param.dtype)
                    freed += n

        # Free SS decoder weights (fully replaced by OVSSDecoder)
        ss_dec = _moduledict_get(self.models, "ss_decoder")
        if ss_dec is not None and isinstance(ss_dec, OVSSDecoder):
            pass  # Already an OV wrapper, no PyTorch params to free

        # Free decoder transformer weights (forward replaced by OV)
        # Keep out_layer and to_representation weights (used in PyTorch)
        for dec_name in ["slat_decoder_gs", "slat_decoder_gs_4", "slat_decoder_mesh"]:
            dec = _moduledict_get(self.models, dec_name)
            if dec is None:
                continue
            kept_prefixes = {"out_layer", "upsample", "mesh_extractor"}
            for name, param in list(dec.named_parameters()):
                top = name.split(".")[0]
                if top not in kept_prefixes:
                    n = param.numel() * param.element_size()
                    param.data = torch.empty(0, dtype=param.dtype)
                    freed += n

        gc.collect()
        if freed > 0:
            print(f"[OV-SAM3D] Freed {freed / (1024**2):.0f} MB of replaced PyTorch parameters")

    # ------------------------------------------------------------------
    #  Internal patching
    # ------------------------------------------------------------------
    def _patch_embedders(self):
        """Replace DINOv2 embedder modules in EmbedderFusers with OV wrappers.

        Uses the single merged DINOv2 backbone (``dino_backbone_ov``) for all
        four embedders, selecting the appropriate output index:
            SS  stage → output 0 (postnorm, prenorm=False)
            SLat stage → output 1 (prenorm,  prenorm=True)
        """
        if "dino_backbone_ov" not in self._compiled:
            # Fall back to per-model keys for backward compatibility
            self._patch_embedders_legacy()
            return

        backbone_compiled = self._compiled["dino_backbone_ov"]

        # SS condition embedder (prenorm=False → output index 0)
        ss_emb = self.condition_embedders["ss_condition_embedder"]
        img_dino = ss_emb.embedder_list[0][0]
        ss_emb.embedder_list[0] = (
            OVDinoEmbedder(backbone_compiled, img_dino.embed_dim,
                           is_mask=False, prenorm=False),
            ss_emb.embedder_list[0][1],
        )
        ss_emb.module_list[0] = nn.Identity()

        mask_dino = ss_emb.embedder_list[1][0]
        ss_emb.embedder_list[1] = (
            OVDinoEmbedder(backbone_compiled, mask_dino.embed_dim,
                           is_mask=True, prenorm=False),
            ss_emb.embedder_list[1][1],
        )
        ss_emb.module_list[1] = nn.Identity()

        # SLat condition embedder (prenorm=True → output index 1)
        slat_emb = self.condition_embedders["slat_condition_embedder"]
        img_dino = slat_emb.embedder_list[0][0]
        slat_emb.embedder_list[0] = (
            OVDinoEmbedder(backbone_compiled, img_dino.embed_dim,
                           is_mask=False, prenorm=True),
            slat_emb.embedder_list[0][1],
        )
        slat_emb.module_list[0] = nn.Identity()

        mask_dino = slat_emb.embedder_list[1][0]
        slat_emb.embedder_list[1] = (
            OVDinoEmbedder(backbone_compiled, mask_dino.embed_dim,
                           is_mask=True, prenorm=True),
            slat_emb.embedder_list[1][1],
        )
        slat_emb.module_list[1] = nn.Identity()

        # Patch projection nets
        for stage_name in ["ss_condition_embedder", "slat_condition_embedder"]:
            emb = self.condition_embedders[stage_name]
            if hasattr(emb, "projection_nets"):
                for i, proj_net in enumerate(emb.projection_nets):
                    proj_key = f"{stage_name}_proj_{i}_ov"
                    if proj_key in self._compiled:
                        emb.projection_nets[i] = OVEmbedderProjection(
                            self._compiled[proj_key]
                        )

        # Restore EmbedderFuser idx_emb from sidecar JSON
        _ef_cfg_path = self._ov_model_dir / "embedder_fuser_config.json"
        if _ef_cfg_path.exists():
            import json as _json
            with open(_ef_cfg_path, "r") as _f:
                _ef_cfg = _json.load(_f)
            for stage_name in ["ss_condition_embedder", "slat_condition_embedder"]:
                key = f"{stage_name}_idx_emb"
                if key in _ef_cfg:
                    emb = self.condition_embedders[stage_name]
                    emb.idx_emb.data = torch.tensor(
                        _ef_cfg[key], dtype=torch.float32
                    )
            print(f"[OV-SAM3D] Loaded EmbedderFuser idx_emb from {_ef_cfg_path}")

        # Patch PointPatchEmbed (SS embedder only)
        if len(ss_emb.embedder_list) > 2:
            ppe = ss_emb.embedder_list[2][0]

            # ── point_proj OV model (replaces nn.Linear + invalid_xyz_token) ──
            ov_point_proj = None
            if "point_proj_ov" in self._compiled:
                ov_point_proj = OVPointProj(self._compiled["point_proj_ov"])
                print("[OV-SAM3D] PointPatchEmbed point_proj → OV")
            else:
                # Fallback: load outer-path weights from JSON into PyTorch
                config_path = self._ov_model_dir / "pointpatch_config.json"
                if config_path.exists():
                    import json as _json
                    with open(config_path, "r") as _f:
                        _outer = _json.load(_f)
                    if "point_proj_weight" in _outer:
                        ppe.point_proj.weight.data = torch.tensor(
                            _outer["point_proj_weight"], dtype=torch.float32
                        )
                    if "point_proj_bias" in _outer:
                        ppe.point_proj.bias.data = torch.tensor(
                            _outer["point_proj_bias"], dtype=torch.float32
                        )
                    if "invalid_xyz_token" in _outer:
                        ppe.invalid_xyz_token.data = torch.tensor(
                            _outer["invalid_xyz_token"], dtype=torch.float32
                        )
                    if "dropped_xyz_token" in _outer and hasattr(ppe, "dropped_xyz_token"):
                        ppe.dropped_xyz_token.data = torch.tensor(
                            _outer["dropped_xyz_token"], dtype=torch.float32
                        )
                    print(f"[OV-SAM3D] Loaded PointPatchEmbed outer weights from {config_path}")

            # Load non-learnable config (remap_type, etc.) from pointpatch_config.json
            config_path = self._ov_model_dir / "pointpatch_config.json"
            if config_path.exists():
                import json as _json
                with open(config_path, "r") as _f:
                    _outer = _json.load(_f)
                if "dropped_xyz_token" in _outer and hasattr(ppe, "dropped_xyz_token"):
                    ppe.dropped_xyz_token.data = torch.tensor(
                        _outer["dropped_xyz_token"], dtype=torch.float32
                    )

            # ── inner attention OV model ──
            ov_inner = None
            if "pointpatch_embed_inner_ov" in self._compiled:
                ov_inner = OVPointPatchEmbedInner(
                    self._compiled["pointpatch_embed_inner_ov"]
                )

            # Monkey-patch the forward
            _ov_pp = ov_point_proj
            _ov_in = ov_inner

            def _ov_ppe_forward(xyz, valid_mask=None, _ppe=ppe, _ov_proj=_ov_pp, _ov_inner=_ov_in):
                with torch.no_grad():
                    xyz_safe = _ppe.resize_input(xyz)
                    if valid_mask is None:
                        valid_mask = xyz_safe.isfinite().all(dim=-1)
                    B, H, W, _ = xyz_safe.shape
                    xyz_safe_clean = xyz_safe.clone()
                    xyz_safe_clean[~valid_mask] = 0.0
                    xyz_remapped = _ppe.point_remapper(xyz_safe_clean)

                if _ov_proj is not None:
                    # OV point_proj: handles projection + invalid token
                    x = _ov_proj(xyz_remapped, valid_mask.float())
                else:
                    # PyTorch fallback
                    x = _ppe.point_proj(xyz_remapped)
                    x[~valid_mask] = 0.0
                    x[~valid_mask] += _ppe.invalid_xyz_token

                if _ov_inner is not None:
                    n_h = torch.tensor(H, dtype=torch.int64)
                    n_w = torch.tensor(W, dtype=torch.int64)
                    return _ov_inner(x, n_h, n_w)
                else:
                    return _ppe.inner_forward(x, B, H, W)

            ppe.forward = _ov_ppe_forward
            if ov_inner is not None:
                print("[OV-SAM3D] PointPatchEmbed inner attention → OV")

        print("[OV-SAM3D] DINOv2 embedders replaced with OV wrappers (merged backbone)")

    def _patch_embedders_legacy(self):
        """Legacy: patch using separate per-model DINOv2 OV files."""
        # SS condition embedder
        ss_emb = self.condition_embedders["ss_condition_embedder"]
        if "ss_dino_image_ov" in self._compiled:
            img_dino = ss_emb.embedder_list[0][0]
            ov_img = OVDinoEmbedder(
                self._compiled["ss_dino_image_ov"],
                embed_dim=img_dino.embed_dim,
                is_mask=False,
                prenorm=False,
            )
            ss_emb.embedder_list[0] = (ov_img, ss_emb.embedder_list[0][1])
            ss_emb.module_list[0] = nn.Identity()

        if "ss_dino_mask_ov" in self._compiled:
            mask_dino = ss_emb.embedder_list[1][0]
            ov_mask = OVDinoEmbedder(
                self._compiled["ss_dino_mask_ov"],
                embed_dim=mask_dino.embed_dim,
                is_mask=True,
                prenorm=False,
            )
            ss_emb.embedder_list[1] = (ov_mask, ss_emb.embedder_list[1][1])
            ss_emb.module_list[1] = nn.Identity()

        # SLat condition embedder
        slat_emb = self.condition_embedders["slat_condition_embedder"]
        if "slat_dino_image_ov" in self._compiled:
            img_dino = slat_emb.embedder_list[0][0]
            ov_img = OVDinoEmbedder(
                self._compiled["slat_dino_image_ov"],
                embed_dim=img_dino.embed_dim,
                is_mask=False,
                prenorm=True,
            )
            slat_emb.embedder_list[0] = (ov_img, slat_emb.embedder_list[0][1])
            slat_emb.module_list[0] = nn.Identity()

        if "slat_dino_mask_ov" in self._compiled:
            mask_dino = slat_emb.embedder_list[1][0]
            ov_mask = OVDinoEmbedder(
                self._compiled["slat_dino_mask_ov"],
                embed_dim=mask_dino.embed_dim,
                is_mask=True,
                prenorm=True,
            )
            slat_emb.embedder_list[1] = (ov_mask, slat_emb.embedder_list[1][1])
            slat_emb.module_list[1] = nn.Identity()

        # Patch projection nets
        for stage_name in ["ss_condition_embedder", "slat_condition_embedder"]:
            emb = self.condition_embedders[stage_name]
            if hasattr(emb, "projection_nets"):
                for i, proj_net in enumerate(emb.projection_nets):
                    proj_key = f"{stage_name}_proj_{i}_ov"
                    if proj_key in self._compiled:
                        emb.projection_nets[i] = OVEmbedderProjection(
                            self._compiled[proj_key]
                        )

        # Restore EmbedderFuser idx_emb from sidecar JSON
        _ef_cfg_path = self._ov_model_dir / "embedder_fuser_config.json"
        if _ef_cfg_path.exists():
            import json as _json
            with open(_ef_cfg_path, "r") as _f:
                _ef_cfg = _json.load(_f)
            for stage_name in ["ss_condition_embedder", "slat_condition_embedder"]:
                key = f"{stage_name}_idx_emb"
                if key in _ef_cfg:
                    emb = self.condition_embedders[stage_name]
                    emb.idx_emb.data = torch.tensor(
                        _ef_cfg[key], dtype=torch.float32
                    )
            print(f"[OV-SAM3D] Loaded EmbedderFuser idx_emb from {_ef_cfg_path}")

        print("[OV-SAM3D] DINOv2 embedders replaced with OV wrappers")

    def _patch_ss_decoder(self):
        """Replace the SS decoder with OV wrapper."""
        if "ss_decoder_ov" in self._compiled:
            ov_dec = OVSSDecoder(self._compiled["ss_decoder_ov"])
            self.models["ss_decoder"] = ov_dec
            print("[OV-SAM3D] SS decoder replaced with OV wrapper")

    def _patch_ss_generator(self):
        """Replace the SS generator backbone forward with OV model."""
        if "ss_generator_ov" not in self._compiled:
            return
        ov_gen = OVSSGenerator(self._compiled["ss_generator_ov"])
        backbone = _unwrap_to_backbone(self.models["ss_generator"])

        # Discover pose latent names from the backbone's config
        pose_names = []
        for _merged, names in backbone.latent_share_transformer.items():
            pose_names = list(names)
            break
        orig_cond_emb = backbone.condition_embedder
        force_zeros = backbone.force_zeros_cond

        # Cache for condition embeddings (inputs never change across ODE steps)
        _ss_cond_cache = {"cond": None, "args_id": None}

        def _ov_forward(latents_dict, t, *cond_args, **cond_kwargs):
            d = cond_kwargs.pop("d", None)
            cfg_activate = cond_kwargs.pop("cfg", False)

            if force_zeros and cfg_activate:
                # CFG uncond branch — zeros
                if _ss_cond_cache["cond"] is not None:
                    cond = torch.zeros_like(_ss_cond_cache["cond"])
                else:
                    cond = orig_cond_emb(*cond_args, **cond_kwargs) * 0
            else:
                # Check cache: condition inputs are identical across ODE steps
                args_id = id(cond_args[0]) if cond_args else None
                if _ss_cond_cache["cond"] is not None and _ss_cond_cache["args_id"] == args_id:
                    cond = _ss_cond_cache["cond"]
                else:
                    cond = orig_cond_emb(*cond_args, **cond_kwargs)
                    _ss_cond_cache["cond"] = cond
                    _ss_cond_cache["args_id"] = args_id

            if d is None:
                d = torch.zeros_like(t)
            shape_latent = latents_dict["shape"]
            pose_list = [latents_dict[name] for name in pose_names]
            results = ov_gen(shape_latent, *pose_list, t, d, cond)
            output = {"shape": results[0]}
            for i, name in enumerate(pose_names):
                output[name] = results[i + 1]
            return output

        backbone.forward = _ov_forward
        self._ov_ss_generator = ov_gen
        print("[OV-SAM3D] SS generator backbone → OV")

    def _patch_slat_generator(self):
        """Replace the SLat generator backbone with OV model.

        **Full OV mode** (``slat_generator_full_ov``):
            The entire backbone (t_embedder + input_layer + input_blocks +
            24 core transformer blocks + out_blocks + out_layer) runs in a
            single OV model.  ALL ~600 M weights are in OV — zero PyTorch
            weight dependencies.

        **Legacy core-only mode** (``slat_generator_core_ov``):
            Only the 24 core transformer blocks run in OV; the U-Net
            input/output blocks (~46 M params) stay in PyTorch.
        """
        # ── Prefer full mode ──
        if "slat_generator_full_ov" in self._compiled:
            ov_full = OVSLatGeneratorFull(self._compiled["slat_generator_full_ov"])
            backbone = _unwrap_to_backbone(self.models["slat_generator"])

            orig_cond_emb = backbone.condition_embedder
            force_zeros = backbone.force_zeros_cond

            # Cache for condition embeddings (inputs never change across ODE steps)
            _slat_cond_cache = {"cond": None, "args_id": None}

            def _ov_forward_full(x, t, *cond_args, **cond_kwargs):
                d = cond_kwargs.pop("d", None)
                if not torch.compiler.is_compiling():
                    if "coords" in cond_kwargs:
                        coords_raw = cond_kwargs.pop("coords")
                    else:
                        coords_raw = cond_args[-1]
                        cond_args = cond_args[:-1]
                else:
                    coords_raw = cond_args[-1]
                    cond_args = cond_args[:-1]
                cfg_activate = cond_kwargs.pop("cfg", False)
                coords = (
                    torch.tensor(coords_raw).to(x.device)
                    if not isinstance(coords_raw, torch.Tensor)
                    else coords_raw
                )

                # Condition embedding with caching
                if force_zeros and cfg_activate:
                    if _slat_cond_cache["cond"] is not None:
                        cond = torch.zeros_like(_slat_cond_cache["cond"])
                    else:
                        cond = orig_cond_emb(*cond_args, **cond_kwargs) * 0
                else:
                    args_id = id(cond_args[0]) if cond_args else None
                    if _slat_cond_cache["cond"] is not None and _slat_cond_cache["args_id"] == args_id:
                        cond = _slat_cond_cache["cond"]
                    else:
                        cond = orig_cond_emb(*cond_args, **cond_kwargs)
                        _slat_cond_cache["cond"] = cond
                        _slat_cond_cache["args_id"] = args_id

                # Full backbone → single OV call
                feats = x[0]                         # (N, C_in)
                coords_xyz = coords[:, 1:].float()   # (N, 3) — strip batch dim
                out_feats = ov_full(feats, coords_xyz, t, cond)
                return out_feats[None]                # (1, N, C_in)

            backbone.forward = _ov_forward_full
            self._ov_slat_generator_full = ov_full
            print("[OV-SAM3D] SLat generator backbone → OV (full, all weights in OV)")
            return

        # ── Fallback: core-only mode ──
        if "slat_generator_core_ov" not in self._compiled:
            return
        ov_core = OVSLatGeneratorCore(self._compiled["slat_generator_core_ov"])
        backbone = _unwrap_to_backbone(self.models["slat_generator"])

        orig_cond_emb = backbone.condition_embedder
        force_zeros = backbone.force_zeros_cond
        bb_dtype = backbone.dtype

        from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor as SPT

        def _ov_forward(x, t, *cond_args, **cond_kwargs):
            d = cond_kwargs.pop("d", None)
            if not torch.compiler.is_compiling():
                if "coords" in cond_kwargs:
                    coords_raw = cond_kwargs.pop("coords")
                else:
                    coords_raw = cond_args[-1]
                    cond_args = cond_args[:-1]
            else:
                coords_raw = cond_args[-1]
                cond_args = cond_args[:-1]
            cfg_activate = cond_kwargs.pop("cfg", False)
            coords = torch.tensor(coords_raw).to(x.device) if not isinstance(coords_raw, torch.Tensor) else coords_raw

            if force_zeros and cfg_activate:
                cond = orig_cond_emb(*cond_args, **cond_kwargs) * 0
            else:
                cond = orig_cond_emb(*cond_args, **cond_kwargs)

            x_sparse = SPT(feats=x[0], coords=coords)
            h = backbone.input_layer(x_sparse).type(bb_dtype)

            t_emb = backbone.t_embedder(t)
            if d is not None and hasattr(backbone, "d_embedder") and backbone.d_embedder is not None:
                t_emb = t_emb + backbone.d_embedder(d)
            if backbone.share_mod:
                t_emb = backbone.adaLN_modulation(t_emb)
            t_emb = t_emb.type(bb_dtype)
            cond = cond.type(bb_dtype)

            skips = []
            for block in backbone.input_blocks:
                h = block(h, t_emb)
                skips.append(h.feats)

            h_feats = h.feats
            if backbone.pe_mode == "ape":
                h_feats = h_feats + backbone.pos_embedder(h.coords[:, 1:]).type(bb_dtype)
            h_feats_out = ov_core(
                h_feats.float(), h.coords[:, 1:].float(),
                t_emb.float(), cond.float(),
            )
            h = h.replace(h_feats_out.type(bb_dtype))

            for block, skip in zip(backbone.out_blocks, reversed(skips)):
                if backbone.use_skip_connection:
                    h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
                else:
                    h = block(h, t_emb)

            h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
            h = backbone.out_layer(h.type(x_sparse.dtype))
            return h.feats[None]

        backbone.forward = _ov_forward
        self._ov_slat_generator_core = ov_core
        print("[OV-SAM3D] SLat generator backbone → OV (core only)")

    def _patch_slat_decoders(self):
        """Replace SLat decoder transformer blocks with OV models."""
        for key, compiled in self._compiled.items():
            if not (key.startswith("slat_decoder_") and key.endswith("_ov")):
                continue
            orig_key = key[:-3]  # e.g. "slat_decoder_gs"
            if orig_key not in self.models:
                continue

            # ── Mesh decoder ──
            if orig_key == "slat_decoder_mesh":
                decoder = self.models[orig_key]
                orig_to_rep = decoder.to_representation

                # ── Prefer merged model (base + upsample in one OV call) ──
                if "slat_decoder_mesh_merged_ov" in self._compiled:
                    ov_merged = OVMeshDecoderMerged(
                        self._compiled["slat_decoder_mesh_merged_ov"]
                    )

                    def _make_ov_mesh_forward_merged(_ov_merged, _to_rep):
                        def _subdivide_coords(coords):
                            offsets = torch.zeros(8, coords.shape[-1], dtype=coords.dtype)
                            for idx in range(8):
                                offsets[idx, 1] = idx // 4
                                offsets[idx, 2] = (idx // 2) % 2
                                offsets[idx, 3] = idx % 2
                            new_c = coords.clone()
                            new_c[:, 1:] *= 2
                            new_c = new_c.unsqueeze(1) + offsets.unsqueeze(0)
                            return new_c.reshape(-1, coords.shape[-1])

                        def _ov_mesh_forward(x_sparse):
                            feats = x_sparse.feats
                            coords = x_sparse.coords
                            coords_xyz = coords[:, 1:].float()
                            out_feats = _ov_merged(feats, coords_xyz)
                            new_coords = coords
                            for _ in range(2):
                                new_coords = _subdivide_coords(new_coords)
                            from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.basic import SparseTensor as SPT
                            h = SPT(out_feats.float(), new_coords.to(coords.dtype))
                            h._scale = x_sparse._scale * 4
                            h._spatial_cache = x_sparse._spatial_cache
                            return _to_rep(h)
                        return _ov_mesh_forward

                    decoder.forward = _make_ov_mesh_forward_merged(
                        ov_merged, orig_to_rep,
                    )
                    self._ov_slat_decoders = getattr(
                        self, "_ov_slat_decoders", {}
                    )
                    self._ov_slat_decoders[orig_key] = ov_merged
                    print(f"[OV-SAM3D] {orig_key} → OV merged (base+upsample) + FlexiCubes")
                    continue

                # ── Fallback: separate base + upsample OV models ──
                ov_mesh_base = OVSLatDecoder(compiled)

                has_ov_upsample = "mesh_decoder_upsample_ov" in self._compiled
                if has_ov_upsample:
                    ov_upsample = OVMeshDecoderUpsample(
                        self._compiled["mesh_decoder_upsample_ov"]
                    )
                else:
                    ov_upsample = None

                orig_upsample = decoder.upsample
                orig_out_layer = decoder.out_layer
                for block in orig_upsample:
                    block.float()
                orig_out_layer.float()

                def _make_ov_mesh_forward(
                    _ov_base, _ov_up, _upsample, _out_layer, _to_rep,
                ):
                    def _subdivide_coords(coords):
                        offsets = torch.zeros(8, coords.shape[-1], dtype=coords.dtype)
                        for idx in range(8):
                            offsets[idx, 1] = idx // 4
                            offsets[idx, 2] = (idx // 2) % 2
                            offsets[idx, 3] = idx % 2
                        new_c = coords.clone()
                        new_c[:, 1:] *= 2
                        new_c = new_c.unsqueeze(1) + offsets.unsqueeze(0)
                        return new_c.reshape(-1, coords.shape[-1])

                    def _ov_mesh_forward(x_sparse):
                        feats = x_sparse.feats
                        coords = x_sparse.coords
                        coords_xyz = coords[:, 1:].float()
                        base_feats = _ov_base(feats, coords_xyz)

                        if _ov_up is not None:
                            out_feats = _ov_up(base_feats)
                            new_coords = coords
                            for _ in range(2):
                                new_coords = _subdivide_coords(new_coords)
                            from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.basic import SparseTensor as SPT
                            h = SPT(out_feats.float(), new_coords.to(coords.dtype))
                            h._scale = x_sparse._scale * 4
                            h._spatial_cache = x_sparse._spatial_cache
                        else:
                            h = x_sparse.replace(base_feats)
                            for block in _upsample:
                                h = block(h)
                            h = h.type(torch.float32)
                            h = _out_layer(h)

                        return _to_rep(h)
                    return _ov_mesh_forward

                decoder.forward = _make_ov_mesh_forward(
                    ov_mesh_base, ov_upsample, orig_upsample,
                    orig_out_layer, orig_to_rep,
                )
                self._ov_slat_decoders = getattr(
                    self, "_ov_slat_decoders", {}
                )
                self._ov_slat_decoders[orig_key] = ov_mesh_base
                if has_ov_upsample:
                    self._ov_slat_decoders["mesh_upsample"] = ov_upsample
                    print(f"[OV-SAM3D] {orig_key} → OV base + OV upsample + FlexiCubes")
                else:
                    print(f"[OV-SAM3D] {orig_key} → OV base + PyTorch upsample/FlexiCubes")
                continue

            # ── GS / GS-4 decoders: full OV replacement ──
            ov_dec = OVSLatDecoder(compiled)
            decoder = self.models[orig_key]
            orig_to_rep = decoder.to_representation

            def _make_ov_decoder_forward(_ov_dec, _orig_to_rep):
                def _ov_forward(x_sparse):
                    feats = x_sparse.feats
                    coords = x_sparse.coords
                    coords_xyz = coords[:, 1:].float()
                    out_feats = _ov_dec(feats, coords_xyz)
                    out_sparse = x_sparse.replace(out_feats)
                    return _orig_to_rep(out_sparse)
                return _ov_forward

            decoder.forward = _make_ov_decoder_forward(ov_dec, orig_to_rep)
            self._ov_slat_decoders = getattr(
                self, "_ov_slat_decoders", {}
            )
            self._ov_slat_decoders[orig_key] = ov_dec
            print(f"[OV-SAM3D] {orig_key} decoder → OV")

    def _patch_moge(self):
        """Replace MoGe depth model with OV model or fall back to PyTorch.

        Prefers OV model (``moge_depth_ov``) if available from conversion.
        Falls back to real PyTorch model on CPU if OV model is not available.
        """
        _pipeline = self

        # Determine which MoGe backend to use
        use_ov = "moge_depth_ov" in self._compiled
        ov_moge = None
        real_moge = None

        if use_ov:
            # Load MoGe config from JSON sidecar
            import json as _json
            moge_config_path = self._ov_model_dir / "moge_config.json"
            moge_config = {}
            if moge_config_path.exists():
                with open(moge_config_path, "r") as _f:
                    moge_config = _json.load(_f)
                print(f"[OV-SAM3D] Loaded MoGe config from {moge_config_path}")
            ov_moge = OVMoGe(self._compiled["moge_depth_ov"], config=moge_config)
        else:
            try:
                real_moge = load_real_moge_cpu()
            except Exception as e:
                print(
                    f"[OV-SAM3D] WARNING: Could not load MoGe model ({e}). "
                    "Falling back to synthetic pointmap data."
                )
                return

        if not use_ov:
            # Replace the depth_model's inner model with the real one
            if hasattr(_pipeline, "depth_model") and _pipeline.depth_model is not None:
                _pipeline.depth_model.model = real_moge
                _pipeline.depth_model.device = torch.device("cpu")

        # Replace compute_pointmap with a version using OV or real MoGe
        _ov_moge_model = ov_moge
        _real_moge_model = real_moge

        def _compute_pointmap_moge(self_pipe, image, pointmap=None):
            loaded_image = self_pipe.image_to_float(image)
            loaded_image = torch.from_numpy(loaded_image)
            loaded_mask = loaded_image[..., -1] if loaded_image.shape[-1] == 4 else torch.ones(loaded_image.shape[:2])
            loaded_image_3ch = loaded_image.permute(2, 0, 1).contiguous()[:3]
            _, H, W = loaded_image_3ch.shape

            if pointmap is None:
                if _ov_moge_model is not None:
                    # OV path: OVMoGe now returns same dict as PT depth_model
                    img_batch = loaded_image_3ch.unsqueeze(0).float()
                    output = _ov_moge_model(img_batch)
                    pointmaps = output["pointmaps"]
                    if pointmaps.dim() == 4:
                        pointmaps = pointmaps[0]  # Remove batch dim → (H, W, 3)
                    intrinsics = output.get("intrinsics", None)
                else:
                    # PyTorch path
                    with torch.no_grad():
                        output = self_pipe.depth_model(loaded_image_3ch)
                    pointmaps = output["pointmaps"]  # (H, W, 3)
                    intrinsics = output.get("intrinsics", None)

                # Apply camera convention transform (using CPU Transform3d)
                try:
                    from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera
                    cam_rot = camera_to_pytorch3d_camera(device="cpu")
                    camera_transform = CPUTransform3d()
                    if hasattr(cam_rot, "rotation"):
                        camera_transform = camera_transform.rotate(cam_rot.rotation)
                    elif hasattr(cam_rot, "get_matrix"):
                        camera_transform = CPUTransform3d(matrix=cam_rot.get_matrix())
                    points_tensor = camera_transform.transform_points(pointmaps)
                except Exception:
                    points_tensor = pointmaps
            else:
                points_tensor = pointmap.to(self_pipe.device)
                if loaded_image_3ch.shape[1:] != points_tensor.shape[:2]:
                    points_tensor = torch.nn.functional.interpolate(
                        points_tensor.permute(2, 0, 1).unsqueeze(0),
                        size=(H, W), mode="nearest",
                    ).squeeze(0).permute(1, 2, 0)
                intrinsics = None

            point_map_tensor = {"pts_color": loaded_image_3ch}

            if intrinsics is None:
                try:
                    from sam3d_objects.pipeline.inference_pipeline_pointmap import (
                        camera_to_pytorch3d_camera,
                        infer_intrinsics_from_pointmap,
                    )
                    cam_rot = camera_to_pytorch3d_camera(device="cpu")
                    camera_transform = CPUTransform3d()
                    if hasattr(cam_rot, "rotation"):
                        camera_transform = camera_transform.rotate(cam_rot.rotation)
                    elif hasattr(cam_rot, "get_matrix"):
                        camera_transform = CPUTransform3d(matrix=cam_rot.get_matrix())
                    points_moge = camera_transform.inverse().transform_points(points_tensor)
                    intrinsics_result = infer_intrinsics_from_pointmap(
                        points_moge, device="cpu"
                    )
                    point_map_tensor["intrinsics"] = intrinsics_result["intrinsics"]
                except Exception:
                    focal = float(max(H, W))
                    point_map_tensor["intrinsics"] = torch.tensor([
                        [focal, 0.0, W / 2.0],
                        [0.0, focal, H / 2.0],
                        [0.0, 0.0, 1.0],
                    ], dtype=torch.float32)
            else:
                point_map_tensor["intrinsics"] = intrinsics

            points_tensor = points_tensor.permute(2, 0, 1)  # (3, H, W)
            if hasattr(self_pipe, "_clip_pointmap"):
                points_tensor = self_pipe._clip_pointmap(points_tensor, loaded_mask)
            point_map_tensor["pointmap"] = points_tensor

            return point_map_tensor

        import types as _types
        _pipeline.compute_pointmap = _types.MethodType(
            _compute_pointmap_moge, _pipeline
        )
        if use_ov:
            self._ov_moge = ov_moge
            print("[OV-SAM3D] MoGe depth model → OV")
        else:
            print("[OV-SAM3D] MoGe depth model → real PyTorch model on CPU")

    def _patch_autocast(self):
        """Patch torch.autocast('cuda') calls in the pipeline to no-ops."""
        import sam3d_objects.pipeline.inference_pipeline as _ip
        import sam3d_objects.pipeline.inference_pipeline_pointmap as _ipm

        # Patch sample_sparse_structure to use CPU-safe autocast
        _orig_ss = self.sample_sparse_structure

        def _sample_ss_cpu(ss_input_dict, inference_steps=None, use_distillation=False):
            # Temporarily replace torch.autocast in the method scope
            orig_autocast = torch.autocast
            torch.autocast = lambda *a, **kw: cpu_autocast()
            try:
                return _orig_ss(ss_input_dict, inference_steps, use_distillation)
            finally:
                torch.autocast = orig_autocast

        self.sample_sparse_structure = _sample_ss_cpu

        # Patch sample_slat
        _orig_slat = self.sample_slat

        def _sample_slat_cpu(slat_input, coords, inference_steps=25, use_distillation=False):
            orig_autocast = torch.autocast
            torch.autocast = lambda *a, **kw: cpu_autocast()
            try:
                return _orig_slat(slat_input, coords, inference_steps, use_distillation)
            finally:
                torch.autocast = orig_autocast

        self.sample_slat = _sample_slat_cpu

        # Patch compute_pointmap (depth model)
        _orig_pm = self.compute_pointmap

        def _compute_pointmap_cpu(image, pointmap=None):
            orig_autocast = torch.autocast
            torch.autocast = lambda *a, **kw: cpu_autocast()
            try:
                return _orig_pm(image, pointmap)
            finally:
                torch.autocast = orig_autocast

        self.compute_pointmap = _compute_pointmap_cpu
        print("[OV-SAM3D] Autocast patched for CPU inference")

    # ------------------------------------------------------------------
    #  Public API — mirrors the original pipeline
    # ------------------------------------------------------------------

    def __call__(
        self,
        image: Union[np.ndarray, "PIL.Image.Image"],
        mask: Optional[Union[None, np.ndarray]] = None,
        seed: Optional[int] = None,
        pointmap=None,
    ) -> dict:
        """
        Callable interface aligned with ``inference.Inference.__call__``.

        Merges *image* and *mask* into an RGBA array and delegates to
        ``self.run()`` — exactly what the original
        ``demo_single_object.ipynb`` does via ``inference(image, mask, seed=42)``.
        """
        mask_uint8 = mask.astype(np.uint8) * 255
        rgba_image = np.concatenate([image[..., :3], mask_uint8[..., None]], axis=-1)
        return self.run(
            image=rgba_image,
            mask=None,
            seed=seed,
            pointmap=pointmap,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
        )


def _ensure_ov_pipeline_class():
    """Dynamically create a class that inherits from
    ``InferencePipelinePointMap`` and has all ``_OVPipelineMixin`` methods.

    Uses ``type()`` to build the class at runtime because CPython forbids
    ``__bases__`` reassignment when the deallocator differs.
    """
    global _ov_pipeline_cls_cache
    try:
        if _ov_pipeline_cls_cache is not None:
            return _ov_pipeline_cls_cache
    except NameError:
        pass

    base_cls = _get_inference_pipeline_pointmap_class()
    # Collect all methods / attrs defined in _OVPipelineMixin (skip dunder noise)
    ns = {
        k: v for k, v in _OVPipelineMixin.__dict__.items()
        if not (k.startswith("__") and k.endswith("__") and k != "__init__" and k != "__call__")
    }
    _ov_pipeline_cls_cache = type("OVInferencePipelinePointMap", (base_cls,), ns)
    return _ov_pipeline_cls_cache

_ov_pipeline_cls_cache = None


def OVInferencePipelinePointMap(*args, **kwargs):
    """Public factory — creates an OV-accelerated pipeline instance.

    On the first call this resolves the real base class
    (``InferencePipelinePointMap``) by calling ``_ensure_ov_pipeline_class()``.
    Subsequent calls reuse the cached class.
    """
    cls = _ensure_ov_pipeline_class()
    return cls(*args, **kwargs)


# ============================================================================
# 6.  FACTORY — one-call pipeline creation
# ============================================================================

def create_ov_pipeline(
    config_path: Union[str, Path],
    ov_model_dir: Union[str, Path] = "./ov_models",
    ov_device: str = "CPU",
    convert: bool = True,
):
    """
    End-to-end helper: load → patch → convert → wrap.

    Parameters
    ----------
    config_path : str | Path
        Path to the ``pipeline.yaml`` (inside the model checkpoint directory).
    ov_model_dir : str | Path
        Where to save / load OpenVINO IR files.
    ov_device : str
        OV device string (``CPU``, ``GPU``, …).
    convert : bool
        If *True*, run model conversion if IR files don't exist yet.

    Returns
    -------
    OVInferencePipelinePointMap
    """
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    config_path = Path(config_path)
    config = OmegaConf.load(config_path)
    config = patch_pipeline_config(config)
    config.workspace_dir = str(config_path.parent)

    ov_model_dir = Path(ov_model_dir)

    if convert:
        print("[OV-SAM3D] Instantiating pipeline on CPU …")
        pipeline = instantiate(config)
        compiled = convert_all_models(pipeline, ov_model_dir, device=ov_device)
        del pipeline  # free heavy weights before creating OV pipeline
    else:
        compiled = load_compiled_models(ov_model_dir, ov_device)

    return OVInferencePipelinePointMap(
        compiled, ov_model_dir=ov_model_dir, config_path=config_path,
    )


def load_compiled_models(
    ov_model_dir: Union[str, Path],
    device: str = "CPU",
) -> Dict[str, ov.CompiledModel]:
    """Load previously converted OV models from disk."""
    ov_model_dir = Path(ov_model_dir)
    core = ov.Core()
    compiled = {}
    model_files = {
        # Merged DINOv2 backbone (replaces 4 separate models)
        "dino_backbone_ov": "dino_backbone.xml",
        # Legacy per-model DINOv2 files (backward compatibility)
        "ss_dino_image_ov": "ss_dino_image.xml",
        "ss_dino_mask_ov": "ss_dino_mask.xml",
        "slat_dino_image_ov": "slat_dino_image.xml",
        "slat_dino_mask_ov": "slat_dino_mask.xml",
        # Other models
        "ss_decoder_ov": "ss_decoder.xml",
        "ss_generator_ov": "ss_generator.xml",
        "slat_generator_full_ov": "slat_generator_full.xml",
        "slat_generator_core_ov": "slat_generator_core.xml",
        "slat_decoder_gs_ov": "slat_decoder_gs.xml",
        "slat_decoder_gs_4_ov": "slat_decoder_gs_4.xml",
        "slat_decoder_mesh_ov": "slat_decoder_mesh.xml",
        "mesh_decoder_upsample_ov": "mesh_decoder_upsample.xml",
        "moge_ov": "moge.xml",
        # PointPatchEmbed inner attention
        "pointpatch_embed_inner_ov": "pointpatch_embed_inner.xml",
        # Merged mesh decoder (base + upsample)
        "slat_decoder_mesh_merged_ov": "slat_decoder_mesh_merged.xml",
    }

    # Also discover any embedder projection / extra decoder files
    for xml_file in ov_model_dir.glob("*.xml"):
        stem = xml_file.stem
        key = f"{stem}_ov"
        if key not in model_files.values() and key not in model_files:
            model_files[key] = xml_file.name
    for key, fname in model_files.items():
        path = ov_model_dir / fname
        if path.exists():
            model = core.read_model(str(path))
            compiled[key] = core.compile_model(model, device)
            print(f"[OV-SAM3D] Loaded {key} from {path}")
        else:
            print(f"[OV-SAM3D] WARNING: {path} not found — skipping {key}")
    return compiled


# ============================================================================
# 7.  UTILITIES
# ============================================================================

def compare_outputs(
    torch_output: torch.Tensor,
    ov_output: torch.Tensor,
    name: str = "output",
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    """
    Compare PyTorch and OV model outputs, print statistics, return pass/fail.
    """
    if isinstance(torch_output, np.ndarray):
        torch_output = torch.from_numpy(torch_output)
    if isinstance(ov_output, np.ndarray):
        ov_output = torch.from_numpy(ov_output)

    torch_output = torch_output.float().detach().cpu()
    ov_output = ov_output.float().detach().cpu()

    abs_diff = (torch_output - ov_output).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    cos_sim = F.cosine_similarity(
        torch_output.flatten().unsqueeze(0),
        ov_output.flatten().unsqueeze(0),
    ).item()

    match = cos_sim > 0.999  # cosine similarity is more robust than allclose
    status = "PASS" if match else "MISMATCH"

    print(
        f"  [{status}] {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
        f"cos_sim={cos_sim:.6f}"
    )
    return match


def load_test_image(
    folder: Optional[Union[str, Path]] = None,
    index: int = 14,
):
    """
    Load a test image + mask from the sam-3d-objects demo data.

    Returns (image_rgba_uint8, mask_bool).
    """
    from PIL import Image

    if folder is None:
        folder = _SAM3D_ROOT / "notebook" / "images" / "shutterstock_stylish_kidsroom_1640806567"
    folder = Path(folder)
    image = np.array(Image.open(str(folder / "image.png"))).astype(np.uint8)
    mask_path = folder / f"{index}.png"
    mask = np.array(Image.open(str(mask_path))).astype(np.uint8) > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    return image, mask


def load_test_masks(
    folder: Optional[Union[str, Path]] = None,
    indices: Optional[List[int]] = None,
    extension: str = ".png",
):
    """
    Load multiple masks from a folder (aligned with ``inference.load_masks``).

    If *indices* is ``None``, discovers all consecutive files ``0.png``, ``1.png``, …
    Returns list of boolean masks.
    """
    from PIL import Image

    if folder is None:
        folder = _SAM3D_ROOT / "notebook" / "images" / "shutterstock_stylish_kidsroom_1640806567"
    folder = Path(folder)

    if indices is None:
        indices = []
        idx = 0
        while (folder / f"{idx}{extension}").exists():
            indices.append(idx)
            idx += 1

    masks = []
    for idx in indices:
        mask_path = folder / f"{idx}{extension}"
        assert mask_path.exists(), f"Mask {mask_path} does not exist"
        mask = np.array(Image.open(str(mask_path))).astype(np.uint8) > 0
        if mask.ndim == 3:
            mask = mask[..., -1]
        masks.append(mask)
    return masks


# ============================================================================
# 8.  VALIDATION — per-model and end-to-end output comparison
# ============================================================================

def validate_all_models(
    pipeline,
    compiled_models: Dict[str, Any],
    ov_model_dir: Union[str, Path] = "./ov_models",
) -> Dict[str, bool]:
    """
    Per-model unit validation: run the same input through PyTorch and OV,
    compare outputs using ``compare_outputs`` (cosine_similarity > 0.999).

    Parameters
    ----------
    pipeline : InferencePipelinePointMap
        The original (un-patched) pipeline on CPU.
    compiled_models : dict
        Dict returned by ``convert_all_models`` or ``load_compiled_models``.
    ov_model_dir : str | Path
        Directory with OV model files and sidecar configs.

    Returns
    -------
    dict
        ``{model_name: bool}`` — True for PASS, False for MISMATCH.
    """
    results = {}
    core = ov.Core()
    ov_model_dir = Path(ov_model_dir)

    print("\n" + "=" * 60)
    print("  Per-Model Validation: PyTorch vs OpenVINO")
    print("=" * 60)

    # ── 1. DINOv2 backbone ────────────────────────────────────────
    if "dino_backbone_ov" in compiled_models:
        print("\n[Validate] DINOv2 backbone …")
        ss_emb = pipeline.condition_embedders["ss_condition_embedder"]
        dino = ss_emb.embedder_list[0][0]
        test_img = torch.randn(1, 3, 518, 518, dtype=torch.float32)
        with torch.no_grad():
            # Use the original DinoBackboneForOV to get PyTorch output
            pt_wrapper = DinoBackboneForOV(dino).eval().float()
            pt_postnorm, pt_prenorm = pt_wrapper(test_img)
        # OV output
        infer_req = compiled_models["dino_backbone_ov"].create_infer_request()
        infer_req.infer(test_img.numpy())
        ov_postnorm = torch.from_numpy(infer_req.get_output_tensor(0).data.copy())
        ov_prenorm = torch.from_numpy(infer_req.get_output_tensor(1).data.copy())
        r1 = compare_outputs(pt_postnorm, ov_postnorm, "DINOv2 postnorm (SS)")
        r2 = compare_outputs(pt_prenorm, ov_prenorm, "DINOv2 prenorm (SLat)")
        results["dino_backbone"] = r1 and r2

    # ── 2. PointPatchEmbed inner ──────────────────────────────────
    if "pointpatch_embed_inner_ov" in compiled_models:
        print("\n[Validate] PointPatchEmbed inner …")
        ss_emb = pipeline.condition_embedders["ss_condition_embedder"]
        if len(ss_emb.embedder_list) > 2:
            ppe = ss_emb.embedder_list[2][0]
            pt_wrapper = PointPatchEmbedInnerForOV(ppe).eval().float()
            H = W = ppe.input_size if hasattr(ppe, "input_size") else 256
            test_x = torch.randn(1, H, W, ppe.embed_dim, dtype=torch.float32)
            test_nh = torch.tensor(H, dtype=torch.int64)
            test_nw = torch.tensor(W, dtype=torch.int64)
            with torch.no_grad():
                pt_out = pt_wrapper(test_x, test_nh, test_nw)
            infer_req = compiled_models["pointpatch_embed_inner_ov"].create_infer_request()
            infer_req.infer([test_x.numpy(), np.array(H, dtype=np.int64), np.array(W, dtype=np.int64)])
            ov_out = torch.from_numpy(infer_req.get_output_tensor(0).data.copy())
            results["pointpatch_embed_inner"] = compare_outputs(pt_out, ov_out, "PointPatchEmbed inner")

    # ── 3. SS decoder ─────────────────────────────────────────────
    if "ss_decoder_ov" in compiled_models:
        print("\n[Validate] SS decoder …")
        ss_dec = pipeline.models["ss_decoder"]
        pt_wrapper = SSDecoderForOV(ss_dec).eval().float()
        test_in = torch.randn(1, 8, 16, 16, 16, dtype=torch.float32)
        with torch.no_grad():
            pt_out = pt_wrapper(test_in)
        infer_req = compiled_models["ss_decoder_ov"].create_infer_request()
        infer_req.infer(test_in.numpy())
        ov_out = torch.from_numpy(infer_req.get_output_tensor(0).data.copy())
        results["ss_decoder"] = compare_outputs(pt_out, ov_out, "SS decoder")

    # ── 4. SS generator ───────────────────────────────────────────
    if "ss_generator_ov" in compiled_models:
        print("\n[Validate] SS generator …")
        ss_gen = pipeline.models["ss_generator"]
        backbone = _unwrap_to_backbone(ss_gen)
        pt_wrapper = SSGeneratorForOV(backbone).eval().float()
        n_latent = 4096
        pose_names = []
        for _merged, names in backbone.latent_share_transformer.items():
            pose_names = list(names)
            break
        # Build example inputs matching SSGeneratorForOV.forward signature
        example_inputs = [torch.randn(1, n_latent, 8, dtype=torch.float32)]  # shape
        for name in pose_names:
            mapping = backbone.latent_mapping[name]
            example_inputs.append(torch.randn(1, 1, mapping.in_channels, dtype=torch.float32))
        example_inputs.append(torch.tensor([0.5], dtype=torch.float32))  # t
        example_inputs.append(torch.tensor([0.0], dtype=torch.float32))  # d
        cond_ch = backbone.cond_channels if hasattr(backbone, "cond_channels") else 1024
        example_inputs.append(torch.randn(1, 1370, cond_ch, dtype=torch.float32))  # cond
        with torch.no_grad():
            pt_out = pt_wrapper(*example_inputs)
        infer_req = compiled_models["ss_generator_ov"].create_infer_request()
        infer_req.infer([x.numpy() for x in example_inputs])
        # Multiple outputs for shape + pose
        ov_out0 = torch.from_numpy(infer_req.get_output_tensor(0).data.copy())
        results["ss_generator"] = compare_outputs(pt_out[0], ov_out0, "SS generator (shape)")

    # ── 5. SLat generator full ────────────────────────────────────
    if "slat_generator_full_ov" in compiled_models:
        print("\n[Validate] SLat generator full …")
        slat_gen = pipeline.models["slat_generator"]
        backbone = _unwrap_to_backbone(slat_gen)
        pt_wrapper = SLatGeneratorFullForOV(backbone).eval().float()
        n_voxels = 512
        in_ch = backbone.in_channels if hasattr(backbone, "in_channels") else 8
        cond_ch = backbone.cond_channels if hasattr(backbone, "cond_channels") else 1024
        test_feats = torch.randn(n_voxels, in_ch, dtype=torch.float32)
        test_coords = torch.randint(0, 32, (n_voxels, 3), dtype=torch.float32)
        test_t = torch.tensor([0.5], dtype=torch.float32)
        test_cond = torch.randn(1, 1370, cond_ch, dtype=torch.float32)
        with torch.no_grad():
            pt_out = pt_wrapper(test_feats, test_coords, test_t, test_cond)
        infer_req = compiled_models["slat_generator_full_ov"].create_infer_request()
        infer_req.infer([test_feats.numpy(), test_coords.numpy(), test_t.numpy(), test_cond.numpy()])
        ov_out = torch.from_numpy(infer_req.get_output_tensor(0).data.copy())
        results["slat_generator_full"] = compare_outputs(pt_out, ov_out, "SLat generator full")

    # ── 6. SLat decoders (GS, GS-4) ──────────────────────────────
    for dkey in ["slat_decoder_gs", "slat_decoder_gs_4"]:
        ov_key = f"{dkey}_ov"
        if ov_key in compiled_models and dkey in pipeline.models:
            print(f"\n[Validate] {dkey} …")
            decoder = pipeline.models[dkey]
            pt_wrapper = SLatDecoderForOV(decoder).eval().float()
            n_voxels = 512
            latent_ch = decoder.in_channels if hasattr(decoder, "in_channels") else 8
            test_feats = torch.randn(n_voxels, latent_ch, dtype=torch.float32)
            test_coords = torch.randint(0, 64, (n_voxels, 3), dtype=torch.float32)
            with torch.no_grad():
                pt_out = pt_wrapper(test_feats, test_coords)
            infer_req = compiled_models[ov_key].create_infer_request()
            infer_req.infer([test_feats.numpy(), test_coords.numpy()])
            ov_out = torch.from_numpy(infer_req.get_output_tensor(0).data.copy())
            results[dkey] = compare_outputs(pt_out, ov_out, dkey)

    # ── 7. Mesh decoder (merged or separate) ──────────────────────
    if "slat_decoder_mesh_merged_ov" in compiled_models and "slat_decoder_mesh" in pipeline.models:
        print("\n[Validate] Mesh decoder (merged) …")
        decoder = pipeline.models["slat_decoder_mesh"]
        pt_wrapper = SLatMeshDecoderMergedForOV(decoder).eval().float()
        n_voxels = 256
        latent_ch = decoder.in_channels if hasattr(decoder, "in_channels") else 8
        test_feats = torch.randn(n_voxels, latent_ch, dtype=torch.float32)
        test_coords = torch.randint(0, 64, (n_voxels, 3), dtype=torch.float32)
        with torch.no_grad():
            pt_out = pt_wrapper(test_feats, test_coords)
        infer_req = compiled_models["slat_decoder_mesh_merged_ov"].create_infer_request()
        infer_req.infer([test_feats.numpy(), test_coords.numpy()])
        ov_out = torch.from_numpy(infer_req.get_output_tensor(0).data.copy())
        results["slat_decoder_mesh_merged"] = compare_outputs(pt_out, ov_out, "Mesh decoder (merged)")

    # ── 8. Embedder projections ───────────────────────────────────
    for key in compiled_models:
        if key.endswith("_proj_0_ov") or key.endswith("_proj_1_ov") or key.endswith("_proj_2_ov"):
            print(f"\n[Validate] {key} …")
            # Find the original projection net
            stage_name = key.replace("_proj_0_ov", "").replace("_proj_1_ov", "").replace("_proj_2_ov", "")
            proj_idx = int(key.split("_proj_")[1].split("_ov")[0])
            if stage_name in pipeline.condition_embedders:
                emb = pipeline.condition_embedders[stage_name]
                if hasattr(emb, "projection_nets") and proj_idx < len(emb.projection_nets):
                    proj_net = emb.projection_nets[proj_idx]
                    pt_wrapper = EmbedderProjectionForOV(proj_net).eval().float()
                    # Detect embed_dim
                    embed_dim = 1024
                    for m in proj_net.modules():
                        if hasattr(m, "normalized_shape") and len(m.normalized_shape) > 0:
                            embed_dim = m.normalized_shape[0]
                            break
                    test_in = torch.randn(1, 100, embed_dim, dtype=torch.float32)
                    with torch.no_grad():
                        pt_out = pt_wrapper(test_in)
                    infer_req = compiled_models[key].create_infer_request()
                    infer_req.infer(test_in.numpy())
                    ov_out = torch.from_numpy(infer_req.get_output_tensor(0).data.copy())
                    results[key] = compare_outputs(pt_out, ov_out, key)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Validation Summary")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:8s} {name}")
        if not passed:
            all_pass = False
    print("-" * 60)
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print("=" * 60 + "\n")
    return results
