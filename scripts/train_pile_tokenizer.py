"""Train a BPE tokenizer on The Pile.

Requires: pip install datasets tokenizers tqdm
Dataset: monology/pile-uncopyrighted on HuggingFace
"""

import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from datasets import load_dataset
from tqdm import tqdm


def train_tokenizer(vocab_size=50000, num_samples=500000, output_path="pile_tokenizer.json"):
    print(f"Loading monology/pile-uncopyrighted (streaming)...")
    dataset = load_dataset("monology/pile-uncopyrighted", split='train', streaming=True) # Put whatever path you have here

    def text_iterator():
        pbar = tqdm(total=num_samples, desc="Sampling")
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
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    print(f"Training with vocab_size={vocab_size}...")
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer, length=num_samples)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    tokenizer.save(output_path)
    print(f"Saved to {output_path}")

    # Quick test
    test = "Hello world! import torch"
    enc = tokenizer.encode(test)
    print(f"Test: {test} -> {len(enc.ids)} tokens")
    print(f"Decoded: {tokenizer.decode(enc.ids)}")

    return tokenizer


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_size", type=int, default=50000)
    p.add_argument("--num_samples", type=int, default=500000)
    p.add_argument("--output", type=str, default="pile_tokenizer.json")
    args = p.parse_args()
    train_tokenizer(args.vocab_size, args.num_samples, args.output)
