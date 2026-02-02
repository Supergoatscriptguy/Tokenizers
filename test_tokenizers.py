"""Test tokenizers. Pass 'all' or a tokenizer name to test."""

import argparse
from pathlib import Path
from tokenizers import Tokenizer

TEST_TEXTS = [
    "Hello world!",
    "The quick brown fox jumps over the lazy dog.",
    "import torch\nimport torch.nn as nn",
    "Artificial intelligence is transforming the world.",
    "1234567890 !@#$%^&*()",
    "The meaning of life is",
]


def test_tokenizer(path):
    name = Path(path).stem
    try:
        tok = Tokenizer.from_file(str(path))
    except Exception as e:
        print(f"  {name}: FAILED to load - {e}")
        return

    print(f"\n=== {name} ===")
    print(f"Vocab size: {tok.get_vocab_size()}")

    total_tokens = 0
    for text in TEST_TEXTS:
        enc = tok.encode(text)
        dec = tok.decode(enc.ids)
        total_tokens += len(enc.ids)
        match = "✓" if dec.strip() == text.strip() else "✗"
        print(f"  [{match}] {len(enc.ids):3d} tokens | {text[:40]}...")

    print(f"  Total: {total_tokens} tokens for {len(TEST_TEXTS)} samples")
    print(f"  Avg: {total_tokens / len(TEST_TEXTS):.1f} tokens/sample")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer", type=str, nargs="?", default="all", help="Tokenizer name (e.g., fineweb_bpe, all)")
    p.add_argument("--dir", type=str, default=".", help="Directory to search")
    args = p.parse_args()

    name = args.tokenizer.lstrip("-")
    search_dir = Path(args.dir)

    if name == "all":
        tokenizers = list(search_dir.glob("*.json"))
        if not tokenizers:
            print(f"No .json tokenizers found in {search_dir}")
            return
        print(f"Found {len(tokenizers)} tokenizers")
        for path in sorted(tokenizers):
            test_tokenizer(path)
    else:
        # Find matching tokenizer
        matches = list(search_dir.glob(f"*{name}*.json"))
        if not matches:
            print(f"No tokenizer matching '{name}' found in {search_dir}")
            print(f"Available: {[p.stem for p in search_dir.glob('*.json')]}")
            return
        for path in matches:
            test_tokenizer(path)


if __name__ == "__main__":
    main()
