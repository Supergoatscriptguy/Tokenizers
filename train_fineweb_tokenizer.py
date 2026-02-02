"""Train a BPE tokenizer on FineWeb."""

import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
import os
os.environ["HF_HOME"] = "E:/fineweb"
os.environ["HF_DATASETS_CACHE"] = "E:/fineweb"
from datasets import load_dataset
from tqdm import tqdm


def train_tokenizer(vocab_size=50257, num_samples=500000, output_path="fineweb_tokenizer.json", cache_dir="E:/fineweb"):
    print(f"Downloading FineWeb sample-10BT to {cache_dir}...")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        cache_dir=cache_dir,
    )

    def text_iterator():
        pbar = tqdm(total=min(num_samples, len(dataset)), desc="Sampling")
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            yield example['text']
            pbar.update(1)
        pbar.close()

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>"],
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    print(f"Training with vocab_size={vocab_size}...")
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer, length=num_samples)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    tokenizer.save(output_path)
    print(f"Saved to {output_path}")

    # Test
    test = "Hello world! The quick brown fox."
    enc = tokenizer.encode(test)
    print(f"Test: {test} -> {len(enc.ids)} tokens")
    print(f"Decoded: {tokenizer.decode(enc.ids)}")

    return tokenizer


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument("--num_samples", type=int, default=500000)
    p.add_argument("--output", type=str, default="fineweb_tokenizer.json")
    p.add_argument("--cache_dir", type=str, default="E:/fineweb")
    args = p.parse_args()
    train_tokenizer(args.vocab_size, args.num_samples, args.output, args.cache_dir)
