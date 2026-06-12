#!/usr/bin/env python3
"""Standalone mobile deploy (no full iterative pipeline)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ab.gpt.iterative_pipeline.mobile.deploy import run_mobile_deploy_for_cycle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description="Export evaluated models to TFLite (mobile track)")
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument(
        "--nneval_dir",
        type=str,
        default=None,
        help="Default: out/curation_output/cycle_N/nneval",
    )
    ap.add_argument("--output_dir", type=str, default="out/curation_output")
    ap.add_argument("--input_size", type=int, default=32)
    ap.add_argument("--max_params", type=int, default=500_000)
    ap.add_argument("--min_accuracy", type=float, default=0.0)
    ap.add_argument("--no_tflite", action="store_true", help="Only copy + weights.pth")
    ap.add_argument("--no_bench", action="store_true", help="Skip desktop TFLite benchmark")
    ap.add_argument("--include_failed", action="store_true", help="Deploy even without 1.json")
    args = ap.parse_args()

    nneval = Path(args.nneval_dir) if args.nneval_dir else Path(args.output_dir) / f"cycle_{args.cycle}" / "nneval"
    summary = run_mobile_deploy_for_cycle(
        args.cycle,
        nneval,
        Path(args.output_dir),
        input_size=args.input_size,
        max_params=args.max_params,
        export_tflite=not args.no_tflite,
        bench_desktop=not args.no_bench,
        min_accuracy=args.min_accuracy,
        only_successful=not args.include_failed,
    )
    logger.info(f"Summary: {summary.get('mobile_root')}/mobile_deploy_summary.json")


if __name__ == "__main__":
    main()
