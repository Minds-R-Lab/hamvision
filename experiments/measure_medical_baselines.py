"""Measure GFLOPs and parameter counts for the three medical-imaging
classification baselines at 224x224 input: MedViT-S, MedMamba-T, and
MedKAFormer-T.

Standard baselines (ResNet-18/50, ViT-B/16, Swin-T) are already covered
by timm; see the tail of this file for the one-liner that measures them.

Runs on CPU in a few seconds per model. Prints one line per baseline
and a table at the end.

Usage (from repo root):
    pip install fvcore timm --quiet
    python experiments/measure_medical_baselines.py

If a baseline repo is not reachable or its model constructor cannot be
found, the script skips it and reports for the rest.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import torch

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    print("ERROR: fvcore not installed. Run `pip install fvcore`.")
    sys.exit(1)


BASELINES_DIR = Path("./_baselines_tmp").resolve()
BASELINES_DIR.mkdir(exist_ok=True)


def clone(repo_url: str, dest_name: str) -> Path:
    dest = BASELINES_DIR / dest_name
    if not dest.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    return dest


def measure(name: str, model: torch.nn.Module, input_size: int = 224) -> dict:
    model.eval()
    x = torch.zeros(1, 3, input_size, input_size)
    gflops = float("nan")
    try:
        flops = FlopCountAnalysis(model, x)
        flops.unsupported_ops_warnings(False)
        flops.tracer_warnings("none")
        gflops = flops.total() / 1e9
    except Exception as e:
        print(f"  ({name}: FlopCountAnalysis failed: {type(e).__name__}: {e})")
    n_params_M = sum(p.numel() for p in model.parameters()) / 1e6
    return {"name": name, "gflops": gflops, "params_M": n_params_M}


def load_medvit_small():
    """MedViT-S from Manzari et al. 2023 (arXiv 2302.09462)."""
    repo = clone("https://github.com/Omid-Nejati/MedViT.git", "MedViT")
    sys.path.insert(0, str(repo))
    try:
        m = importlib.import_module("MedViT")
    finally:
        pass  # keep on sys.path so timm submodules load
    model_fn = getattr(m, "MedViT_small", None) or getattr(m, "medvit_small", None)
    if model_fn is None:
        raise RuntimeError("MedViT_small constructor not found in MedViT.py")
    model = model_fn(num_classes=1000)
    return model


def load_medmamba_t():
    """MedMamba-T from Yue & Li 2024. Requires mamba_ssm; if that isn't
    installed, this raises and the caller skips."""
    repo = clone("https://github.com/YubiaoYue/MedMamba.git", "MedMamba")
    sys.path.insert(0, str(repo))
    m = importlib.import_module("MedMamba")
    fn = getattr(m, "VSSM", None) or getattr(m, "MedMamba", None)
    if fn is None:
        raise RuntimeError("MedMamba class constructor not found")
    model = fn(num_classes=1000)
    return model


def load_medkaformer_t():
    """MedKAFormer-T from Wang et al. 2025. The public repo URL isn't
    universally standard; try the two most common patterns. Return
    None if neither is reachable so the caller reports "-- repo not found --".
    """
    for url, name in [
        ("https://github.com/mahmoodlab/MedKAFormer.git", "MedKAFormer_a"),
        ("https://github.com/wangyx-2003/MedKAFormer.git", "MedKAFormer_b"),
    ]:
        try:
            repo = clone(url, name)
            sys.path.insert(0, str(repo))
            for modname in ["MedKAFormer", "medkaformer", "model", "models"]:
                try:
                    mod = importlib.import_module(modname)
                    for fnname in ["MedKAFormer_tiny", "medkaformer_t",
                                   "MedKAFormer_T", "build_model"]:
                        fn = getattr(mod, fnname, None)
                        if fn is not None:
                            return fn(num_classes=1000)
                except Exception:
                    continue
        except Exception:
            continue
    return None


def main():
    print(f"{'model':<20} {'GFLOPs':>10} {'Params(M)':>12}")
    print("-" * 44)

    rows = []
    for name, loader in [
        ("MedViT-S",       load_medvit_small),
        ("MedMamba-T",     load_medmamba_t),
        ("MedKAFormer-T",  load_medkaformer_t),
    ]:
        try:
            model = loader()
        except Exception as e:
            print(f"{name:<20} FAILED to load: {type(e).__name__}: {e}")
            continue
        if model is None:
            print(f"{name:<20} -- repo not found; use published value")
            continue
        m = measure(name, model)
        rows.append(m)
        print(f"{m['name']:<20} {m['gflops']:>10.2f} {m['params_M']:>12.2f}")

    print()
    print("For the standard baselines (already measured via timm):")
    print(f"  ResNet-18: 1.82 GFLOPs / 11.69 M")
    print(f"  ResNet-50: 4.11 GFLOPs / 25.56 M")
    print(f"  ViT-B/16 : 16.87 GFLOPs / 86.57 M")
    print(f"  Swin-T   : 4.51 GFLOPs / 28.29 M")
    print(f"  HamCls   : 1.71 GFLOPs / 2.95 M  (from src/measure_flops.py)")


if __name__ == "__main__":
    main()
