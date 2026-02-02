# Tokenizers

Tokenizers I trained for my models.

## What's here

| File | Type | Vocab | Trained on |
|------|------|-------|------------|
| `pile_tokenizer.json` | BPE | 50k | The Pile |
| `fineweb_tokenizer.json` | BPE | 50k | FineWeb |
| `fineweb_unigram.json` | Unigram | 50k | FineWeb |
| `fineweb_wordpiece.json` | WordPiece | 50k | FineWeb |

## Which one should I use?

Probably **BPE** or **Unigram**. They handle everything — code, special characters, newlines, etc.

**WordPiece** breaks on newlines and special chars because it uses whitespace tokenization. Fine for BERT-style stuff but not great for code or raw text.

Here's how they compare on some test strings:

| Tokenizer | Avg tokens | Notes |
|-----------|------------|-------|
| pile_tokenizer | 7.7 | Best compression |
| fineweb_bpe | 8.5 | All tests pass |
| fineweb_unigram | 8.5 | All tests pass |
| fineweb_wordpiece | 8.3 | Fails on `\n` and special chars |

## Usage

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("fineweb_tokenizer.json")

tokens = tokenizer.encode("Hello world").ids
text = tokenizer.decode(tokens)
```

## Testing

```bash
# Test all tokenizers
python test_tokenizers.py

# Test specific one
python test_tokenizers.py fineweb_unigram
```

## Training your own

Train all types at once:
```bash
python train_all_tokenizers.py
```

Train specific types:
```bash
python train_all_tokenizers.py --types bpe,unigram
```

Train on The Pile:
```bash
python train_pile_tokenizer.py
```

Train on FineWeb:
```bash
python train_fineweb_tokenizer.py
```

## Install

```bash
pip install tokenizers datasets tqdm
```

## Note

The training scripts have my local paths hardcoded (like `E:/fineweb` for cache). Change them to wherever you want stuff downloaded.
