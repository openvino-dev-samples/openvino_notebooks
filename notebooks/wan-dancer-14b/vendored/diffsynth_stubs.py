# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# stubs for diffsynth deps that aren't usable on a CPU / single-rank OpenVINO env.
# xfuser / yunchang / flash_attn would normally be loaded by the upstream
# `diffsynth/models/wan_video_dit.py` to drive the 8-GPU Ulysses Sequence
# Parallel (USP) attention path. For our single-rank OVIR build, we replace
# each module with a no-op shim that exposes the same public surface.

from __future__ import annotations
import sys as _sys
import types as _types


def _ensure_shim(_module_name: str, _module) -> None:
    """Inject a stub module into sys.modules if it's not already there."""
    if _module_name not in _sys.modules:
        _sys.modules[_module_name] = _module


def install_diffusion_stubs() -> None:
    """Idempotently install xfuser / yunchang / flash_attn shims and force CPU."""
    import os as _os
    # Force PyTorch to stay on CPU. Without this, nn.MultiheadAttention's lazy
    # initialiser probes CUDA and aborts the construction ("Torch not compiled
    # with CUDA enabled"). Setting CUDA_VISIBLE_DEVICES="" is the standard way
    # to disable CUDA detection in CPU-only sub-processes.
    _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _os.environ.setdefault("TORCH_USE_CUDA_DSA", "0")
    # Apply runtime shims for libraries we run on CPU-only.
    _patch_numpy_long()
    _patch_llvmlite_no_deprecation()
    # Belt-and-suspenders: even with the env var, recent PyTorch versions
    # still call into ``torch.cuda._lazy_init()`` from ``torch.empty`` when
    # device is unset. Patch the gate to a no-op so any downstream import
    # that constructs tensors stays on CPU.
    try:
        import torch as _t  # noqa: F401 - imported lazily on purpose
        _t.cuda.is_available = lambda: False  # type: ignore[assignment]
        _t.cuda._lazy_init = lambda: None  # type: ignore[assignment]
        # nn.MultiheadAttention calls ``torch.empty((3*E, E), ...)`` with no
        # dtype/device. Without CUDA we want plain float32 on CPU.
        _orig_empty = _t.empty

        def _cpu_empty(*a, **kw):
            kw.pop("device", None)
            return _orig_empty(*a, **kw)

        _t.empty = _cpu_empty  # type: ignore[assignment]
    except Exception:  # noqa: BLE001
        # If torch isn't importable yet, the env vars above are still in
        # place by the time it is.
        pass

    # ---- xfuser (useless without distributed init, all calls return 0) ------
    if "xfuser" not in _sys.modules:
        _xfuser_dist = _types.ModuleType("xfuser.core.distributed")

        def _zero_rank(*_a, **_kw):
            return 0

        def _one_world(*_a, **_kw):
            return 1

        def _none_group(*_a, **_kw):
            return None

        def _no_op(*_a, **_kw):
            return None

        _xfuser_dist.get_sequence_parallel_rank = _zero_rank
        _xfuser_dist.get_sequence_parallel_world_size = _one_world
        _xfuser_dist.get_ulysses_parallel_world_size = _one_world
        _xfuser_dist.get_ulysses_parallel_rank = _zero_rank
        _xfuser_dist.get_data_parallel_world_size = _one_world
        _xfuser_dist.get_data_parallel_rank = _zero_rank
        _xfuser_dist.get_sp_group = _none_group
        _xfuser_dist.init_distributed_environment = _no_op
        _xfuser_dist.initialize_model_parallel = _no_op

        _xfuser_core = _types.ModuleType("xfuser.core")
        _xfuser_core.distributed = _xfuser_dist

        _xfuser = _types.ModuleType("xfuser")
        _xfuser.core = _xfuser_core

        _ensure_shim("xfuser", _xfuser)
        _ensure_shim("xfuser.core", _xfuser_core)
        _ensure_shim("xfuser.core.distributed", _xfuser_dist)

    # ---- yunchang attention (return identity, optimised out at trace-time) --
    if "yunchang" not in _sys.modules:
        _yunchang = _types.ModuleType("yunchang")

        class _PassthruAttention:  # noqa: D401 - simple shim
            is_paged = False

            def __init__(self, *_, **__):
                pass

            def __call__(self, q, k=None, v=None, *_, **__):
                return q

        _yunchang.LongContextAttention = _PassthruAttention
        _yunchang.ring = _types.ModuleType("yunchang.ring")
        _yunchang.ring.LRringAttention = _PassthruAttention
        _yunchang.ring.RingFlashAttention = _PassthruAttention
        _yunchang.ulysses = _types.ModuleType("yunchang.ulysses")
        _yunchang.ulysses.UlyssesAttention = _PassthruAttention
        _ensure_shim("yunchang", _yunchang)
        _ensure_shim("yunchang.ring", _yunchang.ring)
        _ensure_shim("yunchang.ulysses", _yunchang.ulysses)
        _ensure_shim("yunchang._C", _types.ModuleType("yunchang._C"))

    # ---- flash_attn ---------------------------------------------------------
    # The model file does `import flash_attn` inside try/except, so a missing
    # module would be tolerated anyway. We provide explicit no-ops to keep
    # `torch.compile` traces stable. A bare ``types.ModuleType(...)`` defaults
    # to ``__spec__ == None`` which breaks ``transformers.utils.import_utils``
    # for some optional-dep probes (e.g. phi3). Setting ``__spec__`` to an
    # empty ``ModuleSpec`` is sufficient.
    import importlib.machinery as _machinery

    def _make_fake_spec(name: str) -> _machinery.ModuleSpec:
        return _machinery.ModuleSpec(name=name, loader=None)

    if "flash_attn" not in _sys.modules:
        _fa = _types.ModuleType("flash_attn")
        _fa.__spec__ = _make_fake_spec("flash_attn")
        _fa.__path__ = []  # mark as package so submodules can be requested

        def _fake_attn_func(*_, **__):
            raise RuntimeError(
                "flash_attn is intentionally unavailable in this OV build; the OV "
                "traced DiT uses OpenVINO-native attention."
            )

        _fa.flash_attn_func = _fake_attn_func
        _ensure_shim("flash_attn", _fa)
        # flash_attn_interface also needs a stub module with a non-None __spec__.
        _fai = _types.ModuleType("flash_attn_interface")
        _fai.__spec__ = _make_fake_spec("flash_attn_interface")
        _fai.__path__ = []
        _ensure_shim("flash_attn_interface", _fai)


if __name__ == "__main__":
    install_diffusion_stubs()
    print("Installed DiffSynth stubs: xfuser, yunchang, flash_attn")


def _patch_numpy_long() -> None:
    """``transformers`` T5 modules use ``numpy.long`` for type annotations
    but that attribute disappeared in numpy >= 1.20. Re-add it so the import
    succeeds. (No-op when numpy already exposes ``long``.)
    """
    try:
        import numpy as _np
        if not hasattr(_np, "long"):
            _np.long = _np.int_  # type: ignore[attr-defined]
    except Exception:
        pass


def _patch_llvmlite_no_deprecation() -> None:
    """``llvmlite>=0.46`` deprecated ``binding.initfini.initialize`` and the
    shipped Python function unconditionally raises ``RuntimeError`` — which
    breaks ``numba``'s lazy LLVM inits. Modern LLVM is auto-initialised so a
    no-op stub is safe for our CPU-only flow.
    """
    try:
        import llvmlite.binding as _lb  # noqa: F401 - parent module
        import llvmlite.binding.initfini as _if  # noqa: F401 - submodule

        def _noop_initialize():
            return None

        # Numba resolves ``ll.initialize()`` via the parent module first.
        # Patch every alias we can find so whichever path is used, the
        # no-op is returned instead of the legacy body that raises.
        _if.initialize = _noop_initialize  # type: ignore[assignment]
        if hasattr(_lb, "initialize"):
            _lb.initialize = _noop_initialize  # type: ignore[attr-defined]
        # Some call sites (e.g. ``from llvmlite.binding.initfini import
        # initialize``) bound the legacy name into the importing module's
        # namespace at import time; we cannot reach those without an
        # import hook, but ``lb.initialize`` and ``lb.initfini.initialize``
        # cover the numba call site.
    except Exception:
        pass
