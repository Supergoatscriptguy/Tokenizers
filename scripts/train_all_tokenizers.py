"""Train BPE, Unigram, and WordPiece tokenizers on FineWeb."""

import argparse
from pathlib import Path
import os
os.environ["HF_HOME"] = "E:/fineweb"
os.environ["HF_DATASETS_CACHE"] = "E:/fineweb"
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from tqdm import tqdm


def get_texts(num_samples=500000):
    """Load FineWeb and yield text samples."""
    print("Loading FineWeb...")
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

    texts = []
    pbar = tqdm(total=num_samples, desc="Loading texts")
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        texts.append(example['text'])
        pbar.update(1)
    pbar.close()
    return texts


def train_bpe(texts, vocab_size=50257, output="fineweb_bpe.json"):
    print(f"\n=== Training BPE (vocab={vocab_size}) ===")
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

    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(output)
    print(f"Saved: {output}")
    return tokenizer


def train_unigram(texts, vocab_size=50257, output="fineweb_unigram.json"):
    print(f"\n=== Training Unigram (vocab={vocab_size}) ===")
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(output)
    print(f"Saved: {output}")
    return tokenizer


def train_wordpiece(texts, vocab_size=50257, output="fineweb_wordpiece.json"):
    print(f"\n=== Training WordPiece (vocab={vocab_size}) ===")
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()

    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(output)
    print(f"Saved: {output}")
    return tokenizer


def test_tokenizer(tokenizer, name):
    test = "Hello world! The quick brown fox jumps over the lazy dog."
    enc = tokenizer.encode(test)
    print(f"{name}: {len(enc.ids)} tokens")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument("--num_samples", type=int, default=500000)
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--types", type=str, default="bpe,unigram,wordpiece", help="Comma-separated: bpe,unigram,wordpiece")
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    texts = get_texts(args.num_samples)
    types = args.types.lower().split(",")

    tokenizers = {}

    if "bpe" in types:
        tokenizers["BPE"] = train_bpe(texts, args.vocab_size, f"{args.output_dir}/fineweb_bpe.json")

    if "unigram" in types:
        tokenizers["Unigram"] = train_unigram(texts, args.vocab_size, f"{args.output_dir}/fineweb_unigram.json")

    if "wordpiece" in types:
        tokenizers["WordPiece"] = train_wordpiece(texts, args.vocab_size, f"{args.output_dir}/fineweb_wordpiece.json")

    print("\n=== Results ===")
    for name, tok in tokenizers.items():
        test_tokenizer(tok, name)


if __name__ == "__main__":
    main()
