# NLU Assignment 1: Zipf, Language Models, and HMM POS Tagging

This repository contains my solutions for the first NLU assignment. The work is
split into three parts, all using the Brown corpus from NLTK. The goal is not
only to produce numbers, but also to show what those numbers tell us about
language: word frequencies are very uneven, simple n-gram models struggle with
sparsity, and POS tagging becomes much harder when the word was never seen in
training.

## Project Structure

```text
.
├── nlu_assignment1/
│   ├── part1_zipf.py      # Zipf's law and lexical diversity
│   └── part2_lm.py        # Unigram, bigram, and trigram language models
├── sentences/
│   └── part3_hmm.py       # HMM POS tagger with Viterbi decoding
├── outputs/               # Generated plots and result summaries
├── requirements.txt
└── README.md
```

## Dataset

All three parts use the Brown corpus through NLTK.

- Part 1 uses lower-cased alphabetic Brown tokens for frequency analysis.
- Part 2 uses Brown sentences for train/test language-model evaluation.
- Part 3 uses Brown tagged sentences with the universal POS tagset.

The scripts download the required NLTK data automatically if it is missing:
`brown` and `universal_tagset`.

## Setup

Create or activate a Python environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The NumPy version is pinned to `<2` because some scientific Python packages can
break when a NumPy 2.x version is installed alongside binaries compiled for
NumPy 1.x.

If the NLTK data download fails on first run, install it manually:

```bash
python -m nltk.downloader brown universal_tagset
```

## How to Run

Run the commands from the repository root:

```bash
python nlu_assignment1/part1_zipf.py
python nlu_assignment1/part2_lm.py
python sentences/part3_hmm.py
```

Each script prints the answers to the console. Generated plots and text
summaries are saved in `outputs/`.

## Part 1: Zipf's Law and Corpus Analysis

This part counts word frequencies in the Brown corpus and checks whether the
rank-frequency distribution follows Zipf's law. It also compares raw
type-token ratio (TTR) with moving-average type-token ratio (MATTR).

What the script does:

- prints the top 10 most frequent words;
- fits a line to the log-rank/log-frequency plot;
- prints the top 20 and bottom 20 word types;
- calculates TTR and MATTR for increasing corpus sizes;
- saves two plots:
  - `outputs/part1_zipf_loglog.png`
  - `outputs/part1_ttr.png`

Main observation: frequent words are mostly closed-class function words like
`the`, `of`, `and`, and `to`, while the long tail is made of rare content words.
Raw TTR drops as the corpus grows, but MATTR is more stable because it measures
lexical diversity in fixed-size windows.

## Part 2: N-Gram Language Models

This part builds unigram, bigram, and trigram language models from Brown
sentences. The corpus is split into 80% training data and 20% test data using a
fixed random seed.

The implementation includes:

- vocabulary construction with `<unk>` for rare or unseen words;
- sentence boundary tokens `<s>` and `</s>`;
- add-1 smoothed unigram, bigram, and trigram models;
- an additional add-k bigram model with `k=0.01`;
- perplexity evaluation on the test set;
- simple bigram sentence generation.

Current saved results:

```text
unigram_ppl      = 726.47
bigram_add1_ppl  = 1674.15
bigram_addk_ppl  = 471.68
trigram_add1_ppl = 11843.85
vocabulary_size  = 24738
train_sentences  = 45872
test_sentences   = 11468
```

The interesting point here is that the add-1 bigram and trigram models perform
worse than expected. This is not because context is useless; it is because
Laplace smoothing spreads too much probability mass over a large vocabulary.
The add-k bigram result is much better, which shows that the previous word does
help when the smoothing is less aggressive.

## Part 3: HMM POS Tagger

This part implements a hidden Markov model POS tagger from scratch. NLTK is used
only to load the Brown tagged corpus; the transition probabilities, emission
probabilities, and Viterbi decoder are implemented directly in the script.

The implementation includes:

- add-1 smoothed transition probabilities;
- add-1 smoothed emission probabilities;
- log-space Viterbi decoding;
- overall, seen-word, and OOV accuracy;
- a small analysis of tagging mistakes.

Current saved results:

```text
overall_accuracy = 0.9388
seen_accuracy    = 0.9499
oov_accuracy     = 0.4462
seen_tokens      = 226856
oov_tokens       = 5119
```

The result matches the intuition behind HMM taggers: seen words are usually
tagged well because their emission probabilities are informative. OOV words are
much harder because the model has no direct word-tag evidence and has to depend
mostly on tag transitions.

## Output Files

After running all scripts, the `outputs/` directory should contain:

```text
outputs/part1_zipf_loglog.png
outputs/part1_ttr.png
outputs/part2_results.txt
outputs/part3_results.txt
```

The plot files are useful for the written report, while the text files keep the
main numeric results in a small reproducible form.

## Reproducibility Notes

- Random seed: `1007`
- Train/test split: 80/20 by sentence for Parts 2 and 3
- Corpus: Brown corpus from NLTK
- POS tagset: NLTK universal tagset
- HMM/Viterbi: implemented from scratch

## Troubleshooting

If you see an error like `numpy.core.multiarray failed to import`, reinstall the
requirements:

```bash
python -m pip install -r requirements.txt
```

If you see `Resource brown not found`, download the NLTK data:

```bash
python -m nltk.downloader brown universal_tagset
```

If a command like `python part1_zipf.py` fails, make sure you are using the
paths shown in the "How to Run" section. The Part 1 and Part 2 files are inside
`nlu_assignment1/`, and Part 3 is inside `sentences/`.
