# Tokenizers

Tokenizers I trained for my models.

## Structure

```
fineweb/                     # Trained on FineWeb
  fineweb_tokenizer.json     # BPE
  fineweb_unigram.json       # Unigram
  fineweb_wordpiece.json     # WordPiece

pile/                        # Trained on The Pile
  pile_tokenizer.json        # BPE

scripts/                     # Training and testing
  train_fineweb_tokenizer.py
  train_pile_tokenizer.py
  train_all_tokenizers.py
  train_tokenizer.py
  test_tokenizers.py
```

## Which one should I use?

Probably **BPE** or **Unigram**. They handle everything — code, special characters, newlines, etc.

**WordPiece** breaks on newlines and special chars because it uses whitespace tokenization. Fine for BERT-style stuff but not great for code or raw text.

| Tokenizer | Avg tokens | Notes |
|-----------|------------|-------|
| pile_tokenizer | 7.7 | Best compression |
| fineweb_bpe | 8.5 | All tests pass |
| fineweb_unigram | 8.5 | All tests pass |
| fineweb_wordpiece | 8.3 | Fails on `\n` and special chars |

## Usage

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("fineweb/fineweb_tokenizer.json")

tokens = tokenizer.encode("Hello world").ids
text = tokenizer.decode(tokens)
```

## Testing

```bash
# Test all fineweb tokenizers
python scripts/test_tokenizers.py --dir fineweb

# Test pile tokenizer
python scripts/test_tokenizers.py --dir pile
```

## Training your own

```bash
# Train on FineWeb
python scripts/train_fineweb_tokenizer.py

# Train on The Pile
python scripts/train_pile_tokenizer.py

# Train all types (BPE, Unigram, WordPiece)
python scripts/train_all_tokenizers.py
```

## Install

```bash
pip install tokenizers datasets tqdm
```

## Note

The training scripts have my local paths hardcoded (like `E:/fineweb` for cache). Change them to wherever you want stuff downloaded.
