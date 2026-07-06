"""Part 2: simple n-gram language models on the Brown corpus."""

import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import nltk

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 1007
BOS, EOS, UNK = "<s>", "</s>", "<unk>"


def ensure_data():
    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)


def load_sentences():
    from nltk.corpus import brown

    return [[word.lower() for word in sent] for sent in brown.sents()]


def split_train_test(sentences, train_fraction=0.8):
    random.shuffle(sentences)
    split_at = int(len(sentences) * train_fraction)
    return sentences[:split_at], sentences[split_at:]


def build_vocab(sentences, min_count=2):
    counts = Counter(word for sent in sentences for word in sent)
    vocab = {word for word, count in counts.items() if count >= min_count}
    return vocab | {BOS, EOS, UNK}


def prepare_sentence(sentence, vocab):
    words = [word if word in vocab else UNK for word in sentence]
    return [BOS] + words + [EOS]


def count_ngrams(sentences, vocab):
    unigram = Counter()
    bigram = Counter()
    trigram = Counter()

    for sentence in sentences:
        tokens = prepare_sentence(sentence, vocab)
        unigram.update(tokens)
        bigram.update(zip(tokens, tokens[1:]))
        trigram.update(zip(tokens, tokens[1:], tokens[2:]))

    return unigram, bigram, trigram


def unigram_perplexity(test_sents, unigram, vocab):
    vocab_size = len(vocab)
    token_count = sum(unigram.values())
    log_prob = 0.0
    n = 0

    for sentence in test_sents:
        tokens = prepare_sentence(sentence, vocab)
        for word in tokens[1:]:
            prob = (unigram[word] + 1) / (token_count + vocab_size)
            log_prob += math.log(prob)
            n += 1

    return math.exp(-log_prob / n)


def bigram_perplexity(test_sents, unigram, bigram, vocab, k=1.0):
    vocab_size = len(vocab)
    log_prob = 0.0
    n = 0

    for sentence in test_sents:
        tokens = prepare_sentence(sentence, vocab)
        for word1, word2 in zip(tokens, tokens[1:]):
            prob = (bigram[(word1, word2)] + k) / (unigram[word1] + k * vocab_size)
            log_prob += math.log(prob)
            n += 1

    return math.exp(-log_prob / n)


def trigram_perplexity(test_sents, bigram, trigram, vocab):
    vocab_size = len(vocab)
    log_prob = 0.0
    n = 0

    for sentence in test_sents:
        tokens = [BOS] + prepare_sentence(sentence, vocab)
        for word1, word2, word3 in zip(tokens, tokens[1:], tokens[2:]):
            prob = (trigram[(word1, word2, word3)] + 1) / (bigram[(word1, word2)] + vocab_size)
            log_prob += math.log(prob)
            n += 1

    return math.exp(-log_prob / n)


def build_bigram_choices(bigram):
    choices = defaultdict(list)
    for (word1, word2), count in bigram.items():
        choices[word1].append((word2, count))
    return choices


def sample_sentence(choices, max_len=30):
    sentence = []
    current = BOS

    for _ in range(max_len):
        next_words = choices.get(current)
        if not next_words:
            break

        total = sum(count for _, count in next_words)
        target = random.random() * total
        running = 0

        for word, count in next_words:
            running += count
            if running >= target:
                current = word
                break

        if current == EOS:
            break
        if current != UNK:
            sentence.append(current)

    return " ".join(sentence)


def save_results(results, train_count, test_count, vocab_size):
    with open(OUT / "part2_results.txt", "w") as file:
        file.write(f"unigram_ppl={results['unigram']:.2f}\n")
        file.write(f"bigram_ppl={results['bigram_add1']:.2f}\n")
        file.write(f"bigram_addk_ppl={results['bigram_addk']:.2f}\n")
        file.write(f"trigram_ppl={results['trigram']:.2f}\n")
        file.write(f"vocab={vocab_size}\n")
        file.write(f"train_sents={train_count}\n")
        file.write(f"test_sents={test_count}\n")


def main():
    random.seed(SEED)
    ensure_data()

    sentences = load_sentences()
    train, test = split_train_test(sentences)
    vocab = build_vocab(train)
    unigram, bigram, trigram = count_ngrams(train, vocab)

    results = {
        "unigram": unigram_perplexity(test, unigram, vocab),
        "bigram_add1": bigram_perplexity(test, unigram, bigram, vocab, k=1.0),
        "bigram_addk": bigram_perplexity(test, unigram, bigram, vocab, k=0.01),
        "trigram": trigram_perplexity(test, bigram, trigram, vocab),
    }

    print("Part 2: n-gram language models")
    print(f"Train sentences: {len(train):,}")
    print(f"Test sentences: {len(test):,}")
    print(f"Vocabulary size: {len(vocab):,}\n")

    print("Perplexity")
    print(f"{'Model':<24}{'Value':>10}")
    print(f"{'Unigram add-1':<24}{results['unigram']:>10.2f}")
    print(f"{'Bigram add-1':<24}{results['bigram_add1']:>10.2f}")
    print(f"{'Bigram add-k 0.01':<24}{results['bigram_addk']:>10.2f}")
    print(f"{'Trigram add-1':<24}{results['trigram']:>10.2f}\n")

    print(
        "Add-1 smoothing hurts the bigram and trigram models here because the "
        "vocabulary is large and many contexts are sparse. The smaller add-k "
        "bigram keeps more probability on observed bigrams, so it performs better.\n"
    )

    print("Sample bigram sentences")
    choices = build_bigram_choices(bigram)
    for index in range(1, 6):
        print(f"{index}. {sample_sentence(choices)}")

    save_results(results, len(train), len(test), len(vocab))
    print("\nPart 2 finished. Results saved in outputs/part2_results.txt.")


if __name__ == "__main__":
    main()
