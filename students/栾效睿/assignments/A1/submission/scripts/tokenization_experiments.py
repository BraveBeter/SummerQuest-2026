"""Command-line entry point for the tokenizer experiment suite.

The work is intentionally split by responsibility:

* ``tokenization_data`` preserves document boundaries and writes token arrays.
* ``tokenizer_reports`` runs compression, throughput, and array reports.
* this file only parses the small, documented command-line surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .tokenizer_reports import DEFAULT_CONFIG, repo_path, run_suite
except ImportError:  # pragma: no cover - exercised by the shell entry point.
    from tokenizer_reports import DEFAULT_CONFIG, repo_path, run_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the tokenizer compression, throughput, and encoding reports."
    )
    parser.add_argument("suite", nargs="?", default="suite", choices=["suite"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-compression", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-array-stats", action="store_true")
    parser.add_argument(
        "--encode-arrays",
        action="store_true",
        help="Generate the configured uint16 .npy arrays before reporting their statistics.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a configured input or encoded array is missing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_suite(
        repo_path(args.config),
        repo_path(args.output_dir) if args.output_dir else None,
        run_compression=not args.skip_compression,
        run_throughput=not args.skip_throughput,
        encode_arrays=args.encode_arrays,
        run_array_stats=not args.skip_array_stats,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
