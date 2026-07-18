from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMISSION_ROOT))

from cs336_basics.bpe_tokenizer import BPETokenizer  # noqa: E402
from scripts.tokenization_data import encode_to_npy, iter_documents  # noqa: E402
from scripts.tokenizer_reports import run_suite  # noqa: E402


SPECIAL_TOKEN = "<|endoftext|>"


def byte_tokenizer() -> BPETokenizer:
    return BPETokenizer(
        vocab={index: bytes([index]) for index in range(256)},
        merges=[],
        special_tokens=[SPECIAL_TOKEN],
    )


class TokenizationDataTests(unittest.TestCase):
    def test_eot_delimiters_survive_small_read_chunks(self) -> None:
        text = f"alpha{SPECIAL_TOKEN}牛{SPECIAL_TOKEN}tail"
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.txt"
            input_path.write_bytes(text.encode("utf-8"))

            documents = list(
                iter_documents(input_path, SPECIAL_TOKEN, read_size=5)
            )

        self.assertEqual(
            documents,
            [f"alpha{SPECIAL_TOKEN}", f"牛{SPECIAL_TOKEN}", "tail"],
        )

    def test_streamed_array_matches_encoding_the_complete_input(self) -> None:
        text = f"alpha{SPECIAL_TOKEN}牛{SPECIAL_TOKEN}tail"
        tokenizer = byte_tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "input.txt"
            output_path = directory_path / "tokens.npy"
            input_path.write_text(text, encoding="utf-8")

            stats = encode_to_npy(
                tokenizer,
                input_path,
                output_path,
                SPECIAL_TOKEN,
                batch_bytes=10,
                progress_every_batches=0,
            )
            actual = np.load(output_path)

        expected = np.asarray(tokenizer.encode(text), dtype=np.uint16)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(stats.byte_count, len(text.encode("utf-8")))
        self.assertEqual(stats.token_count, len(expected))

    def test_delimiterless_input_fails_before_its_buffer_grows_unbounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.txt"
            input_path.write_text("x" * 20, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exceeds 8 bytes"):
                list(
                    iter_documents(
                        input_path,
                        SPECIAL_TOKEN,
                        read_size=5,
                        max_document_bytes=8,
                    )
                )

    def test_suite_writes_all_reports_from_one_small_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "input.txt"
            vocab_path = directory_path / "vocab.json"
            merges_path = directory_path / "merges.json"
            output_path = directory_path / "tokens.npy"
            reports_path = directory_path / "reports"
            config_path = directory_path / "config.json"
            text = f"a{SPECIAL_TOKEN}b"
            input_path.write_text(text, encoding="utf-8")
            vocab_path.write_text(json.dumps({"0": "61", "1": "62"}), encoding="utf-8")
            merges_path.write_text("[]", encoding="utf-8")
            config_path.write_text(
                json.dumps(
                    {
                        "special_token": SPECIAL_TOKEN,
                        "seed": 1,
                        "sample_docs": 2,
                        "throughput_target_bytes": 64,
                        "throughput_warmup": 0,
                        "throughput_repeats": 1,
                        "pile_bytes": 64,
                        "output_dir": str(reports_path),
                        "tokenizers": {
                            "toy": {
                                "name": "Toy",
                                "vocab_path": str(vocab_path),
                                "merges_path": str(merges_path),
                            }
                        },
                        "compression": [
                            {
                                "name": "Toy corpus",
                                "input_path": str(input_path),
                                "tokenizers": ["toy"],
                            }
                        ],
                        "throughput": [
                            {
                                "name": "Toy tokenizer",
                                "input_path": str(input_path),
                                "tokenizer": "toy",
                            }
                        ],
                        "encoded_arrays": [
                            {
                                "name": "Toy train",
                                "input_path": str(input_path),
                                "output_path": str(output_path),
                                "tokenizer": "toy",
                            }
                        ],
                        "encoding": {"batch_bytes": 8, "progress_every_batches": 0},
                    }
                ),
                encoding="utf-8",
            )

            run_suite(
                config_path,
                None,
                run_compression=True,
                run_throughput=True,
                encode_arrays=True,
                run_array_stats=True,
                strict=True,
            )

            summary = json.loads((reports_path / "summary.json").read_text(encoding="utf-8"))
            actual = np.load(output_path)
            expected = np.asarray(
                BPETokenizer.from_files(
                    vocab_path,
                    merges_path,
                    special_tokens=[SPECIAL_TOKEN],
                ).encode(text),
                dtype=np.uint16,
            )

        self.assertEqual(summary["compression"][0]["tokens"], len(expected))
        self.assertEqual(summary["encoded_arrays"][0]["status"], "ok")
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
