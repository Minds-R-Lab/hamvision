"""Aggregate V3 paper-protocol run results.

Scans the two default output roots -- outputs_v3_paper for bandpass
(--ablation C) and outputs_v3_paper_lite for HamVision-Lite -- for
`report.txt` files, parses Dice/mIoU/precision/specificity/accuracy per
seed, and prints:
    1. A per-run table (dataset x variant x seed -> metrics).
    2. A per-cell summary (dataset x variant -> mean +/- std).
    3. A LaTeX row snippet ready to paste into Table 4.

Usage (from repo root):
    python experiments/aggregate_v3_paper.py
    python experiments/aggregate_v3_paper.py --roots outputs_v3_paper outputs_v3_paper_lite
    python experiments/aggregate_v3_paper.py --roots /path/to/my/outputs
    python experiments/aggregate_v3_paper.py --ref_full  # print delta vs V2 Table 3 Full HamSeg refs

Layout expected under each root:
    outputs_v3_paper/{dataset}/abl_C/seed_{seed}/report.txt   (bandpass)
    outputs_v3_paper_lite/{dataset}/seed_{seed}/report.txt    (Lite)
    outputs_v3_paper/{dataset}/seed_{seed}/report.txt         (Full HamSeg, if run)

Handles missing files gracefully -- e.g., running with only bandpass ACDC
seed 42 done prints just that one row.

Reference values for Delta vs V2 Table 3 Full HamSeg (per-seed):
    ACDC (4-class, seed 42):   Dice = 93.80
    ISIC 2018 (binary, seed 42): Dice = 90.12
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# V2 paper Table 3 (ablation) per-seed 42 Full HamSeg references (percent).
V2_REF = {
    "acdc": 93.80,
    "isic2018": 90.12,
}

METRICS = ["dice", "miou", "precision", "specificity", "accuracy"]


def parse_report(path: Path) -> Optional[Dict[str, float]]:
    """Parse a HamSeg report.txt file. Returns dict of metric_name -> value*100
    (i.e. percent), or None if the file has no Test Results block."""
    try:
        text = path.read_text()
    except OSError:
        return None
    # Everything between "--- Test Results ---" and the next --- block.
    m = re.search(r"---\s*Test Results\s*---(.*?)(?:---|\Z)", text, re.S)
    if not m:
        return None
    block = m.group(1)
    out: Dict[str, float] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = [p.strip() for p in line.split(":", 1)]
        k = k.lower()
        if k in METRICS:
            try:
                out[k] = float(v) * 100.0  # report is 0-1; we display in pp
            except ValueError:
                pass
    return out or None


def classify_run(report_path: Path, roots: List[Path]) -> Optional[Tuple[str, str, int]]:
    """From a report path, infer (dataset, variant, seed) via the directory
    layout. Returns None if the path doesn't match the expected schema."""
    # Walk relative to whichever root this path is under.
    for root in roots:
        try:
            rel = report_path.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        # Layouts we recognise:
        #   {ds}/abl_{X}/seed_{S}/report.txt   -> variant "abl_X"
        #   {ds}/seed_{S}/report.txt           -> variant depends on root name
        if len(parts) == 4 and parts[1].startswith("abl_") and parts[2].startswith("seed_"):
            ds = parts[0]
            variant = parts[1]  # e.g. abl_C
            seed = int(parts[2].split("_")[-1])
            return ds, variant, seed
        if len(parts) == 3 and parts[1].startswith("seed_"):
            ds = parts[0]
            # Infer variant from the root folder name.
            root_name = root.name.lower()
            if "lite" in root_name:
                variant = "lite"
            elif "paper" in root_name:
                variant = "full"
            else:
                variant = "unknown"
            seed = int(parts[1].split("_")[-1])
            return ds, variant, seed
    return None


def variant_label(v: str) -> str:
    return {
        "abl_A": "ConvNeXt-only (A)",
        "abl_B": "Oscillator-only (B)",
        "abl_C": "Bandpass filterbank",
        "lite":  "HamVision-Lite",
        "full":  "Full HamSeg (Hamiltonian)",
    }.get(v, v)


def fmt_ci(vals: List[float]) -> str:
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return f"{m:.2f} +/- {s:.2f}"


def latex_ci(vals: List[float]) -> str:
    if len(vals) == 1:
        return f"${vals[0]:.2f}$"
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return f"${m:.2f} \\pm {s:.2f}$"


def delta_str(ds: str, mean_dice: float) -> str:
    if ds in V2_REF:
        d = mean_dice - V2_REF[ds]
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.2f} pp"
    return "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="+",
        default=["outputs_v3_paper", "outputs_v3_paper_lite"],
        help="Output root directories to scan.",
    )
    ap.add_argument(
        "--ref_full",
        action="store_true",
        help="Show Delta vs V2 Table 3 Full HamSeg references.",
    )
    args = ap.parse_args()

    roots = [Path(r).resolve() for r in args.roots]
    roots = [r for r in roots if r.is_dir()]
    if not roots:
        print("No valid --roots found. Nothing to aggregate.")
        return 1

    print("== Scanning roots ==")
    for r in roots:
        print(f"  {r}")
    print()

    # (dataset, variant, seed) -> metrics
    runs: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for r in roots:
        for report in r.rglob("report.txt"):
            cls = classify_run(report, roots)
            if cls is None:
                continue
            m = parse_report(report)
            if m is None:
                continue
            runs[cls] = m

    if not runs:
        print("No parseable report.txt files under the given roots.")
        return 1

    # ---------------------------------------------------------------
    # Per-run table.
    # ---------------------------------------------------------------
    print("== Per-run results (Dice / mIoU / Precision / Specificity / Accuracy, all pp) ==")
    header = f"  {'dataset':<10} {'variant':<24} {'seed':<5} " + " ".join(f"{m:>7}" for m in METRICS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for (ds, v, seed) in sorted(runs.keys()):
        m = runs[(ds, v, seed)]
        row = f"  {ds:<10} {variant_label(v):<24} {seed:<5} " + " ".join(
            f"{m.get(k, float('nan')):>7.2f}" for k in METRICS
        )
        print(row)
    print()

    # ---------------------------------------------------------------
    # Per-cell summary.
    # ---------------------------------------------------------------
    cells: Dict[Tuple[str, str], List[Tuple[int, Dict[str, float]]]] = {}
    for (ds, v, seed), m in runs.items():
        cells.setdefault((ds, v), []).append((seed, m))

    print("== Per-cell summary (n seeds -> mean +/- std of Dice, in pp) ==")
    print(f"  {'dataset':<10} {'variant':<24} {'nseeds':<7} {'dice':<20} {'seeds':<20}")
    for (ds, v), entries in sorted(cells.items()):
        dices = [m["dice"] for _, m in entries if "dice" in m]
        seeds = ", ".join(str(s) for s, _ in sorted(entries))
        line = (
            f"  {ds:<10} {variant_label(v):<24} {len(dices):<7} {fmt_ci(dices):<20} {seeds:<20}"
        )
        if args.ref_full and dices:
            line += f"  delta vs V2 Tab.3 Full = {delta_str(ds, statistics.mean(dices))}"
        print(line)
    print()

    # ---------------------------------------------------------------
    # LaTeX snippets ready for Table 4.
    # ---------------------------------------------------------------
    print("== LaTeX rows (paste into manuscript Table 4) ==")
    for (ds, v), entries in sorted(cells.items()):
        dices = [m["dice"] for _, m in entries if "dice" in m]
        if not dices:
            continue
        seed_note = ("(seed~$" + str(entries[0][0]) + "$)") if len(dices) == 1 else f"({len(dices)}-seed)"
        row = (
            f"  % {ds} / {variant_label(v)} {seed_note}\n"
            f"   & \\revC{{{variant_label(v)}}}     & $8.31$--$8.57$~M "
            f"& \\revC{{{latex_ci(dices)}}}  &  \\revC{{...}} \\\\ "
        )
        print(row)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
