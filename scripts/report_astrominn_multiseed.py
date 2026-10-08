#!/usr/bin/env python3
"""Validate and report one or more simulated AstroMiNN AL suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from al_pipeline.astronomy.evaluation.reporting import (
    DEFAULT_METRICS,
    SuiteSpec,
    build_multiseed_report,
    write_report_plots,
    write_report_tables,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate, aggregate, and plot completed multi-seed simulated active-learning suites."
        )
    )
    parser.add_argument(
        "--suite",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "STATE_DIR", "ARTIFACT_DIR"),
        help=(
            "Suite display label followed by its persistent state directory "
            "and heavy-artifact directory. Repeat to compare compatible "
            "training-initialization suites."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory receiving validated tables, figures, and summary.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=("random", "margin", "entropy"),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(135, 246, 357),
    )
    parser.add_argument(
        "--expected-cycles",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("validation", "test", "new_holdout"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "pdf"),
        choices=("png", "pdf", "svg"),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Permit runs whose last completed cycle is below the configured "
            "maximum. Expected strategy/seed directories are still required."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Build the requested report and print its compact inventory."""
    args = parse_args()
    suites = tuple(
        SuiteSpec(
            label=label,
            state_dir=Path(state_dir),
            artifact_dir=Path(artifact_dir),
        )
        for label, state_dir, artifact_dir in args.suite
    )

    report = build_multiseed_report(
        suites,
        expected_strategies=tuple(args.strategies),
        expected_seeds=tuple(args.seeds),
        expected_cycles=args.expected_cycles,
        require_complete=not args.allow_incomplete,
    )

    table_paths = write_report_tables(report, args.output)
    plot_paths = write_report_plots(
        report,
        args.output,
        metrics=tuple(args.metrics),
        splits=tuple(args.splits),
        formats=tuple(args.formats),
    )

    summary = {
        "output": str(args.output.resolve()),
        "validation": report.validation,
        "table_count": len(table_paths),
        "plot_count": len(plot_paths),
        "tables": [str(path) for path in table_paths],
        "plots": [str(path) for path in plot_paths],
    }
    summary_path = args.output / "report_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(summary_path)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
