"""Stream EOT-delimited text into an exact ``uint16`` NumPy token-id array.

This module deliberately owns only data handling and array writing.  Experiment
reporting and command-line parsing live elsewhere.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs336_basics.bpe_tokenizer import BPETokenizer  # noqa: E402


DEFAULT_READ_SIZE = 16 * 1024 * 1024
DEFAULT_BATCH_BYTES = 8 * 1024 * 1024
MAX_DOCUMENT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class EncodingStats:
    byte_count: int
    token_count: int

    @property
    def bytes_per_token(self) -> float:
        return self.byte_count / self.token_count if self.token_count else float("nan")


def iter_documents(
    input_path: Path,
    special_token: str,
    *,
    read_size: int = DEFAULT_READ_SIZE,
    max_document_bytes: int = MAX_DOCUMENT_BYTES,
) -> Iterator[str]:
    """Yield complete documents, retaining each trailing special-token delimiter.

    Keeping the delimiter in the yielded text makes joining adjacent documents
    byte-for-byte equivalent to the original input.  The explicit document cap
    prevents a malformed file without delimiters from silently growing an
    unbounded in-memory buffer.
    """
    delimiter = special_token.encode("utf-8")
    if not delimiter:
        raise ValueError("special_token must not be empty")

    buffer = bytearray()
    with input_path.open("rb") as input_file:
        while chunk := input_file.read(read_size):
            buffer.extend(chunk)
            while (delimiter_start := buffer.find(delimiter)) >= 0:
                boundary = delimiter_start + len(delimiter)
                document = bytes(buffer[:boundary])
                del buffer[:boundary]
                yield document.decode("utf-8")

            if len(buffer) > max_document_bytes:
                raise ValueError(
                    f"Document in {input_path} exceeds {max_document_bytes:,} bytes "
                    f"without {special_token!r}"
                )

    if buffer:
        yield bytes(buffer).decode("utf-8")


def iter_text_batches(
    input_path: Path,
    special_token: str,
    *,
    target_batch_bytes: int,
) -> Iterator[tuple[str, int]]:
    """Yield whole-document batches without changing tokenization boundaries."""
    parts: list[str] = []
    batch_bytes = 0

    for document in iter_documents(input_path, special_token):
        document_bytes = len(document.encode("utf-8"))
        if parts and batch_bytes + document_bytes > target_batch_bytes:
            yield "".join(parts), batch_bytes
            parts = []
            batch_bytes = 0
        parts.append(document)
        batch_bytes += document_bytes

    if parts:
        yield "".join(parts), batch_bytes


def sample_documents(
    input_path: Path,
    special_token: str,
    *,
    sample_size: int,
    seed: int,
) -> list[str]:
    """Return a deterministic reservoir sample of non-empty documents."""
    import random

    rng = random.Random(seed)
    sample: list[str] = []
    seen = 0
    for document in iter_documents(input_path, special_token):
        if document == special_token:
            continue
        seen += 1
        if len(sample) < sample_size:
            sample.append(document)
        elif (replacement := rng.randrange(seen)) < sample_size:
            sample[replacement] = document
    return sample


def compression_stats(tokenizer: BPETokenizer, documents: list[str]) -> EncodingStats:
    byte_count = sum(len(document.encode("utf-8")) for document in documents)
    token_count = sum(len(tokenizer.encode(document)) for document in documents)
    return EncodingStats(byte_count=byte_count, token_count=token_count)


def first_batch(
    input_path: Path,
    special_token: str,
    *,
    target_batch_bytes: int,
) -> tuple[str, int]:
    try:
        return next(
            iter_text_batches(
                input_path,
                special_token,
                target_batch_bytes=target_batch_bytes,
            )
        )
    except StopIteration as exc:
        raise ValueError(f"No text found in {input_path}") from exc


def encode_to_npy(
    tokenizer: BPETokenizer,
    input_path: Path,
    output_path: Path,
    special_token: str,
    *,
    batch_bytes: int = DEFAULT_BATCH_BYTES,
    progress_every_batches: int = 10,
) -> EncodingStats:
    """Encode an EOT-delimited corpus in two streaming passes.

    ``.npy`` needs its final shape before it can be memory-mapped.  Counting in
    the first pass avoids materializing billions of Python integers; the second
    pass writes each batch directly to its final position.
    """
    _validate_uint16_vocab(tokenizer)
    total = _count_tokens(
        tokenizer,
        input_path,
        special_token,
        batch_bytes,
        progress_every_batches,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_array = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint16,
        shape=(total.token_count,),
    )

    offset = 0
    for batch_index, (text, _) in enumerate(
        iter_text_batches(input_path, special_token, target_batch_bytes=batch_bytes),
        start=1,
    ):
        token_ids = tokenizer.encode(text)
        next_offset = offset + len(token_ids)
        if next_offset > total.token_count:
            raise RuntimeError("Input changed while tokenizing; refusing a truncated array")
        token_array[offset:next_offset] = np.asarray(token_ids, dtype=np.uint16)
        offset = next_offset
        if progress_every_batches and batch_index % progress_every_batches == 0:
            print(f"wrote batch {batch_index}: tokens={offset}", flush=True)

    if offset != total.token_count:
        raise RuntimeError(
            "Input changed while tokenizing; token count differs between the two passes"
        )
    token_array.flush()
    return total


def _validate_uint16_vocab(tokenizer: BPETokenizer) -> None:
    token_ids = tokenizer.vocab.keys()
    if min(token_ids, default=0) < 0 or max(token_ids, default=0) > np.iinfo(np.uint16).max:
        raise ValueError("Tokenizer IDs must fit in uint16")


def _count_tokens(
    tokenizer: BPETokenizer,
    input_path: Path,
    special_token: str,
    batch_bytes: int,
    progress_every_batches: int,
) -> EncodingStats:
    byte_count = 0
    token_count = 0
    for batch_index, (text, current_bytes) in enumerate(
        iter_text_batches(input_path, special_token, target_batch_bytes=batch_bytes),
        start=1,
    ):
        byte_count += current_bytes
        token_count += len(tokenizer.encode(text))
        if progress_every_batches and batch_index % progress_every_batches == 0:
            print(
                f"counted batch {batch_index}: bytes={byte_count} tokens={token_count}",
                flush=True,
            )
    return EncodingStats(byte_count=byte_count, token_count=token_count)
