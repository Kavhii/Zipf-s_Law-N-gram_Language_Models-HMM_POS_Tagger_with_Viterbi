"""Part 1: Zipf's law and lexical diversity on the Brown corpus."""

import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np


def ensure_data():
    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)


def load_tokens():
    from nltk.corpus import brown

    return [word.lower() for word in brown.words() if re.search(r"[a-z]", word.lower())]


def print_top_words(tokens):
    freq = Counter(tokens)
    ranked = freq.most_common()

    print("Q1. Most frequent words")
    print(f"{'Rank':<6}{'Word':<15}{'Count':>10}")
    for rank, (word, count) in enumerate(ranked[:10], start=1):
        print(f"{rank:<6}{word:<15}{count:>10,}")

    print(f"\nTotal tokens: {len(tokens):,}")
    print(f"Vocabulary size: {len(freq):,}\n")
    return ranked


def plot_zipf(ranked):
    ranks = np.arange(1, len(ranked) + 1)
    freqs = np.array([count for _, count in ranked], dtype=float)
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(freqs)
    slope, intercept = np.polyfit(log_ranks, log_freqs, 1)

    plt.figure(figsize=(8, 5.5))
    plt.scatter(log_ranks, log_freqs, s=6, alpha=0.35, label="word types")
    plt.plot(log_ranks, slope * log_ranks + intercept, lw=2, label=f"fit: slope={slope:.3f}")
    plt.plot(log_ranks, -log_ranks + intercept, lw=1.5, ls="--", label="ideal Zipf slope")
    plt.xlabel("log10(rank)")
    plt.ylabel("log10(frequency)")
    plt.title("Zipf's Law on the Brown Corpus")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "part1_zipf_loglog.png", dpi=150)
    plt.close()

    print("Q2. Zipf fit")
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print("A perfect Zipf distribution would have a slope close to -1.")
    print("This corpus is close in shape, but not a perfect fit.\n")
    return slope, intercept


def print_top_and_bottom(ranked):
    top = ", ".join(f"{word} ({count})" for word, count in ranked[:20])
    bottom = ", ".join(f"{word} ({count})" for word, count in ranked[-20:])

    print("Q3. Head and tail of the vocabulary")
    print(f"Top 20: {top}\n")
    print(f"Bottom 20: {bottom}\n")
    print(
        "The most frequent words are mostly function words such as determiners, "
        "prepositions, and conjunctions. The rare end is mostly content words, "
        "many of which appear only once.\n"
    )


def ttr(tokens):
    return len(set(tokens)) / len(tokens)


def mattr(tokens, window=1000):
    if len(tokens) <= window:
        return ttr(tokens)

    step = max(1, (len(tokens) - window) // 2000)
    ratios = []
    for start in range(0, len(tokens) - window + 1, step):
        window_tokens = tokens[start:start + window]
        ratios.append(len(set(window_tokens)) / window)
    return float(np.mean(ratios))


def analyze_lexical_diversity(tokens):
    sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    sizes = [size for size in sizes if size <= len(tokens)]
    ttr_values = []
    mattr_values = []

    print("Q4/Q5. TTR and MATTR")
    print(f"{'Tokens':>10}{'TTR':>12}{'MATTR':>12}")
    for size in sizes:
        sample = tokens[:size]
        raw_ttr = ttr(sample)
        moving_ttr = mattr(sample)
        ttr_values.append(raw_ttr)
        mattr_values.append(moving_ttr)
        print(f"{size:>10,}{raw_ttr:>12.4f}{moving_ttr:>12.4f}")

    plt.figure(figsize=(8, 5.5))
    plt.plot(sizes, ttr_values, "o-", lw=2, label="TTR")
    plt.plot(sizes, mattr_values, "s--", lw=2, label="MATTR, window=1000")
    plt.xlabel("Corpus size in tokens")
    plt.ylabel("Lexical diversity")
    plt.title("TTR and MATTR Across Corpus Sizes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style="plain", axis="x")
    plt.tight_layout()
    plt.savefig(OUT / "part1_ttr.png", dpi=150)
    plt.close()

    print(
        "\nRaw TTR falls as the corpus grows because repeated words become more "
        "common. MATTR is more stable because it compares fixed-size windows.\n"
    )


def main():
    ensure_data()
    tokens = load_tokens()
    ranked = print_top_words(tokens)
    plot_zipf(ranked)
    print_top_and_bottom(ranked)
    analyze_lexical_diversity(tokens)
    print("Part 1 finished. Plots saved in outputs/.")


if __name__ == "__main__":
    main()
