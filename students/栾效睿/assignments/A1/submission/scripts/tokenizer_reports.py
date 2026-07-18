"""Reproducible tokenizer experiment reports built on ``tokenization_data``."""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs336_basics.bpe_tokenizer import BPETokenizer  # noqa: E402
try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .tokenization_data import (
        DEFAULT_BATCH_BYTES,
        EncodingStats,
        compression_stats,
        encode_to_npy,
        first_batch,
        sample_documents,
    )
except ImportError:  # pragma: no cover - exercised by the shell entry point.
    from tokenization_data import (
        DEFAULT_BATCH_BYTES,
        EncodingStats,
        compression_stats,
        encode_to_npy,
        first_batch,
        sample_documents,
    )

DEFAULT_CONFIG = ROOT / "configs" / "tokenizer_experiments.json"


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    vocab_path: Path
    merges_path: Path


def run_suite(
    config_path: Path,
    output_dir: Path | None,
    *,
    run_compression: bool,
    run_throughput: bool,
    encode_arrays: bool,
    run_array_stats: bool,
    strict: bool,
) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    destination = output_dir or repo_path(config["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "config_path": rel_path(config_path),
        "output_dir": rel_path(destination),
        "compression": None,
        "throughput": None,
        "encoded_arrays": None,
    }
    if run_compression:
        summary["compression"] = compression_report(config, destination)
    if run_throughput:
        summary["throughput"] = throughput_report(config, destination)
    if encode_arrays:
        encode_configured_arrays(config, destination)
    if run_array_stats:
        summary["encoded_arrays"] = encoded_array_report(config, destination, strict=strict)

    write_json(destination / "summary.json", summary)
    print(f"wrote {rel_path(destination / 'summary.json')}", flush=True)


def compression_report(config: dict[str, Any], output_dir: Path) -> list[dict[str, object]]:
    special_token = config["special_token"]
    required = [repo_path(item["input_path"]) for item in config["compression"]]
    required.extend(_all_tokenizer_paths(config))
    require_existing(required)

    rows: list[dict[str, object]] = []
    for experiment in config["compression"]:
        input_path = repo_path(experiment["input_path"])
        documents = sample_documents(
            input_path,
            special_token,
            sample_size=int(config["sample_docs"]),
            seed=int(config["seed"]),
        )
        if not documents:
            raise ValueError(f"No documents were sampled from {rel_path(input_path)}")

        for tokenizer_key in experiment["tokenizers"]:
            spec = tokenizer_spec(config, tokenizer_key)
            stats = compression_stats(load_tokenizer(spec, special_token), documents)
            row: dict[str, object] = {
                "dataset": experiment["name"],
                "input_path": rel_path(input_path),
                "tokenizer": spec.name,
                "sample_docs": len(documents),
                "seed": config["seed"],
                "bytes": stats.byte_count,
                "tokens": stats.token_count,
                "bytes_per_token": stats.bytes_per_token,
            }
            rows.append(row)
            print(
                f"ratio {experiment['name']} / {spec.name}: "
                f"bytes={stats.byte_count} tokens={stats.token_count} "
                f"bytes/token={stats.bytes_per_token:.4f}",
                flush=True,
            )

    write_json(output_dir / "compression_ratio.json", rows)
    return rows


def throughput_report(config: dict[str, Any], output_dir: Path) -> list[dict[str, object]]:
    special_token = config["special_token"]
    target_bytes = int(config["throughput_target_bytes"])
    warmup = int(config["throughput_warmup"])
    repeats = int(config["throughput_repeats"])
    pile_bytes = int(config["pile_bytes"])
    required = [repo_path(item["input_path"]) for item in config["throughput"]]
    required.extend(_all_tokenizer_paths(config))
    require_existing(required)

    rows: list[dict[str, object]] = []
    for experiment in config["throughput"]:
        input_path = repo_path(experiment["input_path"])
        spec = tokenizer_spec(config, experiment["tokenizer"])
        tokenizer = load_tokenizer(spec, special_token)
        text, byte_count = first_batch(
            input_path,
            special_token,
            target_batch_bytes=target_bytes,
        )

        for _ in range(warmup):
            tokenizer.encode(text)

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            token_count = len(tokenizer.encode(text))
            times.append(time.perf_counter() - start)

        median_seconds = statistics.median(times)
        bytes_per_second = byte_count / median_seconds
        row: dict[str, object] = {
            "name": experiment["name"],
            "input_path": rel_path(input_path),
            "tokenizer": spec.name,
            "target_bytes": target_bytes,
            "bytes": byte_count,
            "tokens": token_count,
            "warmup": warmup,
            "repeats": repeats,
            "times_sec": times,
            "median_sec": median_seconds,
            "mb_per_sec": bytes_per_second / 1_000_000,
            "pile_bytes": pile_bytes,
            "pile_serial_hours": pile_bytes / bytes_per_second / 3600,
        }
        rows.append(row)
        print(
            f"throughput {experiment['name']}: bytes={byte_count} "
            f"median={median_seconds:.4f}s speed={row['mb_per_sec']:.3f} MB/s",
            flush=True,
        )

    write_json(output_dir / "throughput.json", rows)
    return rows


def encode_configured_arrays(config: dict[str, Any], output_dir: Path) -> None:
    special_token = config["special_token"]
    encoding = config.get("encoding", {})
    batch_bytes = int(encoding.get("batch_bytes", DEFAULT_BATCH_BYTES))
    progress_every_batches = int(encoding.get("progress_every_batches", 10))

    for item in config["encoded_arrays"]:
        spec = tokenizer_spec(config, item["tokenizer"])
        input_path = repo_path(item["input_path"])
        output_path = repo_path(item["output_path"])
        require_existing([input_path, spec.vocab_path, spec.merges_path])
        print(f"encode {item['name']} -> {rel_path(output_path)}", flush=True)
        stats = encode_to_npy(
            load_tokenizer(spec, special_token),
            input_path,
            output_path,
            special_token,
            batch_bytes=batch_bytes,
            progress_every_batches=progress_every_batches,
        )
        write_json(
            output_dir / f"{item['output_path'].replace('/', '_')}.summary.json",
            encoding_summary(input_path, output_path, spec, stats, batch_bytes),
        )


def encoded_array_report(
    config: dict[str, Any], output_dir: Path, *, strict: bool
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in config["encoded_arrays"]:
        input_path = repo_path(item["input_path"])
        output_path = repo_path(item["output_path"])
        row: dict[str, object] = {
            "name": item["name"],
            "input_path": rel_path(input_path),
            "output_path": rel_path(output_path),
            "tokenizer": config["tokenizers"][item["tokenizer"]]["name"],
        }
        missing = next((path for path in (output_path, input_path) if not path.exists()), None)
        if missing:
            row["status"] = "missing_encoded_array" if missing == output_path else "missing_raw_input"
            rows.append(row)
            if strict:
                raise FileNotFoundError(f"Missing required file: {rel_path(missing)}")
            print(f"array-stats skip {item['name']}: missing {rel_path(missing)}", flush=True)
            continue

        token_ids = np.load(output_path, mmap_mode="r")
        token_count = int(token_ids.shape[0])
        byte_count = input_path.stat().st_size
        row.update(
            {
                "status": "ok",
                "dtype": str(token_ids.dtype),
                "tokens": token_count,
                "bytes": byte_count,
                "bytes_per_token": byte_count / token_count if token_count else float("nan"),
            }
        )
        rows.append(row)
        print(
            f"array-stats {item['name']}: tokens={token_count} "
            f"bytes/token={row['bytes_per_token']:.4f}",
            flush=True,
        )

    write_json(output_dir / "encoded_arrays.json", rows)
    return rows


def encoding_summary(
    input_path: Path,
    output_path: Path,
    spec: TokenizerSpec,
    stats: EncodingStats,
    batch_bytes: int,
) -> dict[str, object]:
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "tokenizer": spec.name,
        "vocab_path": str(spec.vocab_path),
        "merges_path": str(spec.merges_path),
        "dtype": "uint16",
        "tokens": stats.token_count,
        "bytes": stats.byte_count,
        "bytes_per_token": stats.bytes_per_token,
        "batch_bytes": batch_bytes,
        "workers": 1,
    }


def tokenizer_spec(config: dict[str, Any], key: str) -> TokenizerSpec:
    item = config["tokenizers"][key]
    return TokenizerSpec(
        name=item["name"],
        vocab_path=repo_path(item["vocab_path"]),
        merges_path=repo_path(item["merges_path"]),
    )


def load_tokenizer(spec: TokenizerSpec, special_token: str) -> BPETokenizer:
    return BPETokenizer.from_files(
        spec.vocab_path,
        spec.merges_path,
        special_tokens=[special_token],
    )


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def require_existing(paths: list[Path]) -> None:
    missing = [rel_path(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _all_tokenizer_paths(config: dict[str, Any]) -> list[Path]:
    return [
        path
        for item in config["tokenizers"].values()
        for path in (repo_path(item["vocab_path"]), repo_path(item["merges_path"]))
    ]
