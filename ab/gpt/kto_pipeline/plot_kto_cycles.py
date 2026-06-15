#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402

# Model-card reference (ABrain/NNGPT-UniqueArch-Rag, 1-epoch CIFAR-10).
CARD_BEST = 0.6398
CARD_AVG = 0.5099


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n. Returns (lo, hi) in [0,1]."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _parse_cycle(c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Flatten one cycle record (from the aggregate OR a per-cycle metrics.json)."""
    if "bucketing" not in c:
        return None  # failed cycle with no metrics
    b = c.get("bucketing", {})
    bd = b.get("undesirable_breakdown", {})
    gen = int(c.get("generated", 0)) or 1
    new_des = int(b.get("new_desirable", 0))
    not_novel = int(bd.get("not_novel", 0))
    return {
        "cycle": int(c.get("cycle", 0)),
        "generated": gen,
        "evaluated": int(b.get("evaluated_accuracies", gen)),
        "best": float(b.get("best_accuracy", 0.0) or 0.0),
        "avg": float(b.get("avg_accuracy", 0.0) or 0.0),
        "new_desirable": new_des,
        "new_undesirable": int(b.get("new_undesirable", 0)),
        "not_novel": not_novel,
        "low_accuracy": int(bd.get("low_accuracy", 0)),
        "non_compiling": int(bd.get("non_compiling", 0)),
        "runtime_error": int(bd.get("runtime_error", 0)),
        "unparseable": int(bd.get("unparseable", 0)),
        "desirable_total": int(b.get("desirable_total", 0)),
        "undesirable_total": int(b.get("undesirable_total", 0)),
        "pass_acc": new_des + not_novel,  # cleared the accuracy bar (novel + duplicate)
        "trained": bool(c.get("training", {}).get("success", False)),
    }


def load_cycles(results_path: Path, cycles_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """
    Build the per-cycle list, keyed by cycle number, from:
      1. the aggregate all_cycles_results.json (current session's cycles), then
      2. every cycle_<n>/metrics.json found under cycles_dir — these PERSIST across
         resumes, so they give the FULL history even when the aggregate is partial.
    """
    by_cycle: Dict[int, Dict[str, Any]] = {}

    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        for c in data.get("cycles", []):
            rec = _parse_cycle(c)
            if rec is not None:
                by_cycle[rec["cycle"]] = rec
    except Exception as e:  # noqa: BLE001
        print(f"[plot][warn] could not read aggregate {results_path}: {e}")

    if cycles_dir and cycles_dir.exists():
        for mfile in sorted(cycles_dir.glob("cycle_*/metrics.json")):
            try:
                rec = _parse_cycle(json.loads(mfile.read_text(encoding="utf-8")))
                if rec is not None:
                    by_cycle[rec["cycle"]] = rec  # authoritative per-cycle metrics
            except Exception:  # noqa: BLE001
                continue

    return [by_cycle[k] for k in sorted(by_cycle)]


def load_per_model_accuracies(cycles_dir: Optional[Path]) -> Dict[int, List[float]]:
    """
    Best-effort: scan cycle_<n>/nneval/*/eval_info.json for per-model accuracies
    so we can compute median + quality-threshold curves.  Returns {} if not found.
    """
    accs: Dict[int, List[float]] = {}
    if not cycles_dir or not cycles_dir.exists():
        return accs
    for cycle_dir in sorted(cycles_dir.glob("cycle_*")):
        try:
            n = int(cycle_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        nneval = cycle_dir / "nneval"
        if not nneval.exists():
            continue
        vals: List[float] = []
        for info in nneval.glob("*/eval_info.json"):
            try:
                payload = json.loads(info.read_text(encoding="utf-8"))
                res = payload.get("eval_results", {})
                a = res.get("accuracy", res.get("acc"))
                if a is not None:
                    vals.append(float(a))
            except Exception:  # noqa: BLE001
                continue
        if vals:
            accs[n] = vals
    return accs


def plot_overview(cycles: List[Dict[str, Any]], per_model: Dict[int, List[float]],
                  out_dir: Path, threshold: float) -> Path:
    xs = [r["cycle"] for r in cycles]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Self-Contained KTO — Iterative Cycle Analysis", fontsize=15, fontweight="bold")

    # Panel A: accuracy trends (best, average, +median if available) + card refs
    a = ax[0][0]
    a.plot(xs, [r["best"] * 100 for r in cycles], "o-", color="#1f77b4", label="Best")
    a.plot(xs, [r["avg"] * 100 for r in cycles], "s-", color="#ff7f0e", label="Average")
    if per_model:
        med_x, med_y = [], []
        for r in cycles:
            v = per_model.get(r["cycle"])
            if v:
                s = sorted(v)
                med = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2
                med_x.append(r["cycle"]); med_y.append(med * 100)
        if med_x:
            a.plot(med_x, med_y, "^-", color="#2ca02c", label="Median")
    a.axhline(CARD_BEST * 100, ls="--", color="#1f77b4", alpha=0.5, label=f"Card best ({CARD_BEST*100:.1f}%)")
    a.axhline(CARD_AVG * 100, ls="--", color="#ff7f0e", alpha=0.5, label=f"Card avg ({CARD_AVG*100:.1f}%)")
    a.axhline(threshold * 100, ls=":", color="gray", alpha=0.7, label=f"Threshold ({threshold*100:.0f}%)")
    a.set_title("First-Epoch Accuracy Trends"); a.set_xlabel("Cycle"); a.set_ylabel("Accuracy (%)")
    a.grid(True, alpha=0.3); a.legend(fontsize=8)

    # Panel B: generation outcome rates per cycle
    b = ax[0][1]
    b.plot(xs, [100 * r["evaluated"] / r["generated"] for r in cycles], "P-",
           color="#17becf", label="Valid (compiled+trained)")
    b.plot(xs, [100 * r["new_desirable"] / r["generated"] for r in cycles], "o-",
           color="#2ca02c", label="Desirable (pass+novel)")
    b.plot(xs, [100 * r["pass_acc"] / r["generated"] for r in cycles], "s-",
           color="#9467bd", label="Cleared accuracy bar")
    b.plot(xs, [100 * r["low_accuracy"] / r["generated"] for r in cycles], "v-",
           color="#d62728", label="Below threshold")
    b.plot(xs, [100 * r["not_novel"] / r["generated"] for r in cycles], "d-",
           color="#8c564b", label="Not novel (duplicate)")
    b.set_title("Generation Outcomes (% of generated)"); b.set_xlabel("Cycle")
    b.set_ylabel("% of models"); b.grid(True, alpha=0.3); b.legend(fontsize=8)

    # Panel C: per-cycle counts (desirable vs undesirable)
    c = ax[1][0]
    width = 0.4
    c.bar([x - width / 2 for x in xs], [r["new_desirable"] for r in cycles], width,
          label="New desirable", color="#2ca02c")
    c.bar([x + width / 2 for x in xs], [r["new_undesirable"] for r in cycles], width,
          label="New undesirable", color="#d62728", alpha=0.8)
    c.set_title("Per-Cycle Bucketing Counts"); c.set_xlabel("Cycle"); c.set_ylabel("Count")
    c.grid(True, alpha=0.3, axis="y"); c.legend(fontsize=9)

    # Panel D: cumulative training-data growth
    d = ax[1][1]
    d.plot(xs, [r["desirable_total"] for r in cycles], "o-", color="#2ca02c", label="Desirable total")
    d.plot(xs, [r["undesirable_total"] for r in cycles], "s-", color="#d62728", label="Undesirable total")
    d.plot(xs, [r["desirable_total"] + r["undesirable_total"] for r in cycles], "^-",
           color="#1f77b4", label="Total accumulated")
    d.set_title("Preference-Data Growth"); d.set_xlabel("Cycle"); d.set_ylabel("Cumulative examples")
    d.grid(True, alpha=0.3); d.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / "kto_cycle_overview.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_success_ci(cycles: List[Dict[str, Any]], out_dir: Path, threshold: float) -> Path:
    """≥threshold accuracy rate per cycle with Wilson 95% CI (over evaluated models)."""
    xs, rate, lo, hi = [], [], [], []
    for r in cycles:
        n = r["evaluated"] or r["generated"]
        k = r["pass_acc"]
        p = k / n if n else 0.0
        l, h = wilson_ci(k, n)
        xs.append(r["cycle"]); rate.append(p * 100)
        lo.append((p - l) * 100); hi.append((h - p) * 100)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.errorbar(xs, rate, yerr=[lo, hi], fmt="o-", color="#1f77b4", capsize=3, ecolor="#1f77b4")
    ax.set_title(f"Accuracy-Pass Rate per Cycle (≥{threshold*100:.0f}%) with 95% CI")
    ax.set_xlabel("Cycle"); ax.set_ylabel("Models clearing threshold (%)")
    ax.grid(True, alpha=0.3); ax.set_ylim(0, 100)
    fig.tight_layout()
    path = out_dir / "kto_success_rate_ci.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def save_csv(cycles: List[Dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "kto_cycle_summary.csv"
    cols = ["cycle", "generated", "evaluated", "best", "avg", "new_desirable",
            "new_undesirable", "low_accuracy", "not_novel", "desirable_total",
            "undesirable_total", "trained"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in cycles:
            w.writerow(r)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Plot KTO per-cycle analysis")
    p.add_argument("--results", type=str, default="all_cycles_results.json",
                   help="Path to the KTO all_cycles_results.json")
    p.add_argument("--out_dir", type=str, default="kto_plots")
    p.add_argument("--threshold", type=float, default=None,
                   help="Accuracy threshold (default: read from the json's accuracy_threshold)")
    p.add_argument("--cycles_dir", type=str, default=None,
                   help="Optional dir with cycle_<n>/nneval/*/eval_info.json for median/quality curves")
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    threshold = args.threshold
    if threshold is None:
        try:
            threshold = float(json.loads(results_path.read_text()).get("accuracy_threshold", 0.40))
        except Exception:  # noqa: BLE001
            threshold = 0.40

    # Auto-detect cycles_dir next to the results file if not given (gives full
    # history via per-cycle metrics.json + per-model accuracies for median curves).
    cycles_dir = Path(args.cycles_dir) if args.cycles_dir else results_path.parent
    cycles = load_cycles(results_path, cycles_dir)
    if not cycles:
        raise SystemExit("No cycles with metrics found in the results file.")
    per_model = load_per_model_accuracies(cycles_dir)

    f1 = plot_overview(cycles, per_model, out_dir, threshold)
    f2 = plot_success_ci(cycles, out_dir, threshold)
    f3 = save_csv(cycles, out_dir)

    # Console summary (handy for the report).
    best_overall = max(cycles, key=lambda r: r["best"])
    print("=" * 64)
    print(f"KTO run: {len(cycles)} cycles ({cycles[0]['cycle']}..{cycles[-1]['cycle']})")
    print(f"  best accuracy this run : {best_overall['best']*100:.2f}%  (cycle {best_overall['cycle']})")
    print(f"  card reference best    : {CARD_BEST*100:.2f}%")
    print(f"  last-cycle avg accuracy: {cycles[-1]['avg']*100:.2f}%  (card avg {CARD_AVG*100:.2f}%)")
    print(f"  desirable accumulated  : {cycles[-1]['desirable_total']}")
    print(f"  per-model eval data    : {'found' if per_model else 'not found (median/quality skipped)'}")
    print("-" * 64)
    print(f"  figure : {f1}")
    print(f"  figure : {f2}")
    print(f"  csv    : {f3}")
    print("=" * 64)


if __name__ == "__main__":
    main()
