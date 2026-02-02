"""Train a BPE tokenizer from preprocessed .npy shards or text files."""

import argparse
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tqdm import tqdm


def train_from_shards(data_dir, vocab_size=50257, output_path="tokenizer.json",
                      num_shards=50, existing_tokenizer=None):
    """Train tokenizer by decoding tokens from .npy shards back to text."""

    if existing_tokenizer is None:
        raise ValueError("Need existing tokenizer to decode shards. Pass --decode_with")

    # Load existing tokenizer to decode
    decoder_tok = Tokenizer.from_file(existing_tokenizer)

    shard_files = sorted(Path(data_dir).glob("shard_*.npy"))[:num_shards]
    print(f"Using {len(shard_files)} shards")

    def text_iterator():
        for shard_path in tqdm(shard_files, desc="Processing shards"):
            data = np.load(shard_path)
            for seq in data[:1000]:  # Sample 1000 sequences per shard
                text = decoder_tok.decode(seq.tolist())
                if text.strip():
                    yield text

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|padding|>"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    print(f"Training tokenizer with vocab_size={vocab_size}...")
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(output_path)
    print(f"Saved to {output_path}")
    return tokenizer


def train_from_text(data_dir, vocab_size=50257, output_path="tokenizer.json", pattern="*.txt"):
    """Train tokenizer directly from text files."""

    files = list(Path(data_dir).glob(pattern))
    if not files:
        files = list(Path(data_dir).glob("**/*.txt"))
    print(f"Found {len(files)} text files")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|padding|>"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    print(f"Training with vocab_size={vocab_size}...")
    tokenizer.train([str(f) for f in files], trainer)
    tokenizer.save(output_path)
    print(f"Saved to {output_path}")
    return tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument("--output", type=str, default="tokenizer.json")
    p.add_argument("--mode", type=str, choices=["text", "shards"], default="text")
    p.add_argument("--num_shards", type=int, default=50, help="Number of shards to use (shards mode)")
    p.add_argument("--decode_with", type=str, default=None, help="Existing tokenizer to decode shards")
    p.add_argument("--pattern", type=str, default="*.txt", help="File pattern (text mode)")
    args = p.parse_args()

    if args.mode == "shards":
        tokenizer = train_from_shards(args.data_dir, args.vocab_size, args.output,
                                       args.num_shards, args.decode_with)
    else:
        tokenizer = train_from_text(args.data_dir, args.vocab_size, args.output, args.pattern)

    # Test
    test = "Hello world! The quick brown fox."
    enc = tokenizer.encode(test)
    print(f"\nTest: {test}")
    print(f"Tokens: {enc.ids}")
    print(f"Decoded: {tokenizer.decode(enc.ids)}")


if __name__ == "__main__":
    main()
