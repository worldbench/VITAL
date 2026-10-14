"""Aggregate per-scene scorer components into NAVSIM v1 PDMS."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from vital.evaluation.pdms import pdms_components


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute PDMS from scorer component CSV")
    parser.add_argument("--scores", type=Path, required=True, help="per-scene component CSV")
    parser.add_argument("--output", type=Path, required=True, help="summary JSON")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = []
    with args.scores.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("valid", "True").lower() in {"false", "0"}:
                raise ValueError('Invalid scenes must be fixed, not silently excluded')
            if row.get('token') == 'average':
                continue
            rows.append(pdms_components(row))
    if not rows:
        raise ValueError("No valid score rows found")
    keys = list(rows[0])
    means = {key: sum(row[key] for row in rows if not math.isnan(row[key])) /
             sum(not math.isnan(row[key]) for row in rows) for key in keys}
    result = {"format": "anonymous_pdms_summary_v1", "num_valid_scenes": len(rows),
              "mean": means,
              "protocol": "NAVSIM v1 PDM aggregation; components are official scorer outputs"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PDMS={means['PDMS']:.4f} over {len(rows)} valid scenes")


if __name__ == "__main__":  # pragma: no cover
    main()
