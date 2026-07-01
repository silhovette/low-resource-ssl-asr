"""Train a character-level n-gram KenLM language model from LibriSpeech transcripts.

Usage
-----
  python scripts/train_lm.py --manifest data/manifests/train-clean-100_1h_hf.jsonl \\
                             --output lm/char_5gram.arpa --order 5

If you have access to the full LibriSpeech LM corpus text, you can train a
stronger word-level LM instead:

  python scripts/train_lm.py --corpus /path/to/librispeech-lm-norm.txt \\
                             --output lm/word_4gram.arpa --order 4 --word-level
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def iter_texts_from_manifest(manifest_path: str):
    """Yield one normalised transcript per line from a JSONL manifest."""
    from src.utils import read_jsonl
    for row in read_jsonl(manifest_path):
        text = row.get("text_norm", row.get("text", ""))
        if text.strip():
            yield text.strip()


def preprocess_char(text: str) -> str:
    """Lower-case and insert spaces between characters for char-level LM."""
    text = text.lower()
    # Insert space between every character
    return " ".join(list(text))


def preprocess_word(text: str) -> str:
    """Lower-case, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Train a KenLM n-gram LM")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=str,
                     help="Path to a JSONL manifest with 'text_norm' or 'text' fields")
    src.add_argument("--corpus", type=str,
                     help="Path to a plain-text corpus (one sentence per line)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .arpa file path")
    parser.add_argument("--order", type=int, default=5,
                        help="n-gram order (default: 5)")
    parser.add_argument("--word-level", action="store_true",
                        help="Train a word-level LM (default: char-level)")
    parser.add_argument("--tmp-dir", type=str, default="/tmp/kenlm_train",
                        help="Temp directory for intermediate files")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)

    # ---- collect text ----
    preprocess = preprocess_word if args.word_level else preprocess_char
    tmp_text = Path(args.tmp_dir) / "corpus.txt"
    count = 0
    with tmp_text.open("w", encoding="utf-8") as f:
        if args.manifest:
            for text in iter_texts_from_manifest(args.manifest):
                processed = preprocess(text)
                if processed:
                    f.write(processed + "\n")
                    count += 1
        else:
            with open(args.corpus, "r", encoding="utf-8") as fc:
                for line in fc:
                    processed = preprocess(line)
                    if processed:
                        f.write(processed + "\n")
                        count += 1
    print(f"Wrote {count} lines to {tmp_text}")

    # ---- train KenLM ----
    try:
        import kenlm  # noqa: F401 — just check availability
    except ImportError:
        print("ERROR: kenlm is required. Install with:")
        print("  pip install https://github.com/kpu/kenlm/archive/master.zip")
        sys.exit(1)

    import subprocess
    arpa_path = args.output
    bin_path = tmp_text.with_suffix(".arpa")
    # kenlm's `lmplz` builds the model; `build_binary` makes a .binary file
    cmd = [
        "lmplz",
        "-o", str(args.order),
        "--text", str(tmp_text),
        "--arpa", str(bin_path),
        "--discount_fallback",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Rename
    bin_path.rename(arpa_path)
    print(f"LM saved to {arpa_path}")
    print(f"  size: {Path(arpa_path).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
