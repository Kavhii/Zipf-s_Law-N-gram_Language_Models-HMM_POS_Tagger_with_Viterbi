"""Part 3: HMM POS tagger with Viterbi decoding."""

import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import nltk

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 1007
BOS, EOS = "<s>", "</s>"


def ensure_data():
    packages = {"brown": "corpora/brown", "universal_tagset": "taggers/universal_tagset"}
    for package, path in packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


def load_tagged_sentences():
    from nltk.corpus import brown

    return [[(word.lower(), tag) for word, tag in sent] for sent in brown.tagged_sents(tagset="universal")]


def split_train_test(sentences, train_fraction=0.8):
    random.shuffle(sentences)
    split_at = int(len(sentences) * train_fraction)
    return sentences[:split_at], sentences[split_at:]


def train_hmm(sentences):
    tags = set()
    vocab = set()
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tag_counts = Counter()

    for sentence in sentences:
        previous = BOS
        for word, tag in sentence:
            tags.add(tag)
            vocab.add(word)
            transitions[previous][tag] += 1
            emissions[tag][word] += 1
            tag_counts[tag] += 1
            previous = tag
        transitions[previous][EOS] += 1

    tag_list = sorted(tags)
    tag_count = len(tag_list)
    vocab_size = len(vocab)

    def transition_prob(prev_tag, tag):
        return (transitions[prev_tag][tag] + 1) / (sum(transitions[prev_tag].values()) + tag_count + 1)

    def emission_prob(tag, word):
        return (emissions[tag][word] + 1) / (tag_counts[tag] + vocab_size + 1)

    return {
        "tags": tag_list,
        "vocab": vocab,
        "transitions": transitions,
        "emissions": emissions,
        "tag_counts": tag_counts,
        "tag_count": tag_count,
        "vocab_size": vocab_size,
        "transition_prob": transition_prob,
        "emission_prob": emission_prob,
    }


def viterbi(words, model):
    tags = model["tags"]
    transition_prob = model["transition_prob"]
    emission_prob = model["emission_prob"]

    if not words:
        return []

    scores = []
    backpointers = []

    first_scores = {}
    first_backpointers = {}
    for tag in tags:
        first_scores[tag] = math.log(transition_prob(BOS, tag)) + math.log(emission_prob(tag, words[0]))
        first_backpointers[tag] = None
    scores.append(first_scores)
    backpointers.append(first_backpointers)

    for i in range(1, len(words)):
        word_scores = {}
        word_backpointers = {}
        for tag in tags:
            best_prev = None
            best_score = float("-inf")
            emit_score = math.log(emission_prob(tag, words[i]))

            for prev_tag in tags:
                score = scores[i - 1][prev_tag] + math.log(transition_prob(prev_tag, tag)) + emit_score
                if score > best_score:
                    best_score = score
                    best_prev = prev_tag

            word_scores[tag] = best_score
            word_backpointers[tag] = best_prev

        scores.append(word_scores)
        backpointers.append(word_backpointers)

    best_last_tag = max(tags, key=lambda tag: scores[-1][tag] + math.log(transition_prob(tag, EOS)))
    path = [best_last_tag]

    for i in range(len(words) - 1, 0, -1):
        path.append(backpointers[i][path[-1]])

    return list(reversed(path))


def evaluate(sentences, model):
    total = seen_total = oov_total = 0
    correct = seen_correct = oov_correct = 0
    errors = []

    for sentence in sentences:
        words = [word for word, _ in sentence]
        gold_tags = [tag for _, tag in sentence]
        predicted_tags = viterbi(words, model)

        for word, gold, predicted in zip(words, gold_tags, predicted_tags):
            seen = word in model["vocab"]
            total += 1
            seen_total += int(seen)
            oov_total += int(not seen)

            if gold == predicted:
                correct += 1
                seen_correct += int(seen)
                oov_correct += int(not seen)
            else:
                errors.append((word, gold, predicted, seen))

    return {
        "overall_acc": correct / total,
        "seen_acc": seen_correct / seen_total if seen_total else 0.0,
        "oov_acc": oov_correct / oov_total if oov_total else 0.0,
        "seen_total": seen_total,
        "oov_total": oov_total,
        "errors": errors,
    }


def print_model_examples(model):
    transitions = model["transitions"]
    emissions = model["emissions"]

    noun_total = sum(transitions["NOUN"].values()) + model["tag_count"] + 1
    verb_total = model["tag_counts"]["VERB"] + model["vocab_size"] + 1

    print("Common HMM probabilities")
    print("Top transitions after NOUN:")
    for tag, count in transitions["NOUN"].most_common(5):
        print(f"  NOUN -> {tag:<6} {(count + 1) / noun_total:.4f}")

    print("\nTop words emitted by VERB:")
    for word, count in emissions["VERB"].most_common(5):
        print(f"  VERB -> {word:<12} {(count + 1) / verb_total:.4f}")
    print()


def error_reason(word, seen):
    ambiguous_words = {"that", "to", "as", "like", "back", "well", "up", "down", "one", "s"}
    if not seen:
        return "unseen word"
    if word in ambiguous_words:
        return "common ambiguous word"
    return "rare word or sparse context"


def print_evaluation(results):
    print("Tagging accuracy")
    print(f"{'Group':<14}{'Tokens':>10}{'Accuracy':>12}")
    print(f"{'Overall':<14}{results['seen_total'] + results['oov_total']:>10,}{results['overall_acc']:>12.4f}")
    print(f"{'Seen':<14}{results['seen_total']:>10,}{results['seen_acc']:>12.4f}")
    print(f"{'OOV':<14}{results['oov_total']:>10,}{results['oov_acc']:>12.4f}\n")

    print("A few tagging mistakes")
    print(f"{'Word':<16}{'Gold':<8}{'Pred':<8}Reason")
    for word, gold, predicted, seen in results["errors"][:10]:
        print(f"{word:<16}{gold:<8}{predicted:<8}{error_reason(word, seen)}")

    oov_errors = sum(1 for _, _, _, seen in results["errors"] if not seen)
    seen_errors = len(results["errors"]) - oov_errors
    print(
        f"\nSeen words create more total errors ({seen_errors:,}), but OOV words "
        f"are much less reliable. Their accuracy is only {results['oov_acc'] * 100:.1f}%.\n"
    )


def save_results(results):
    with open(OUT / "part3_results.txt", "w") as file:
        file.write(f"overall_acc={results['overall_acc']:.4f}\n")
        file.write(f"seen_acc={results['seen_acc']:.4f}\n")
        file.write(f"oov_acc={results['oov_acc']:.4f}\n")
        file.write(f"seen_tokens={results['seen_total']}\n")
        file.write(f"oov_tokens={results['oov_total']}\n")


def main():
    random.seed(SEED)
    ensure_data()

    sentences = load_tagged_sentences()
    train, test = split_train_test(sentences)
    model = train_hmm(train)
    results = evaluate(test, model)

    print("Part 3: HMM POS tagger")
    print(f"Train sentences: {len(train):,}")
    print(f"Test sentences: {len(test):,}\n")

    print_model_examples(model)
    print_evaluation(results)
    save_results(results)
    print("Part 3 finished. Results saved in outputs/part3_results.txt.")


if __name__ == "__main__":
    main()
