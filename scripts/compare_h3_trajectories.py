#!/usr/bin/env python3
"""Compare completed Exact/Grid H3 50-cycle trajectory receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from h3_a100.trajectory import compare_trajectory_roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-cycles", type=int, default=50)
    parser.add_argument("--expected-world-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = compare_trajectory_roots(
        args.reference,
        args.candidate,
        expected_cycles=args.expected_cycles,
        expected_world_size=args.expected_world_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison evidence: {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
