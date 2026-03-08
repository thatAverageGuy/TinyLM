"""
Naive BPE Tokenizer
===================
A from-scratch implementation of Byte Pair Encoding as described in
the original Sennrich et al. 2015 paper (https://arxiv.org/abs/1508.07909).

BPE Tokenization Algorithm (Simplified):
-----------------------------------------
- Input Raw Text: "This text is raw text"

- Convert raw text into individual characters spaced with a word boundary
  notifier: {"T h i s </w>": 1, "t e x t </w>": 2, "i s </w>": 1, "r a w </w>": 1}

- Count bigram pairs and return a bigram pair freq dict: {(T, h): 1, (i, s): 2, ...}

- Find max pair in bigram pairs: e.g. (i, s)

- Merge (i, s) into a single entity (is): e.g post merge
  {"T h is </w>": 1, "t e x t </w>": 2, "is </w>": 1, "r a w </w>": 1}
  Add this merging to a merge_rules file.

- Repeat "n_merges" times. "n_merges" is the only hyperparam in this algorithm.

- Post n_merges, save all the merge pairs in exact order as they were performed
  in merge_rules.txt. Each line contains the symbols that were merged, in sequence.

- Also store the final vocab: all merged tokens + each individual character from
  the training corpus, assigned simple integer token IDs.
  e.g. {'[UNK]': 0, '[PAD]': 1, '[BOS]': 2, '[EOS]': 3, 'T': 4, 'h': 5, ...}

Note: Special tokens [PAD], [UNK], [EOS], [BOS] are kept with the lowest token IDs
      so they can be easily identified and their positions remain fixed even if
      the vocab grows.

Note: This implementation is case-sensitive ('T' and 't' are separate tokens)
      and uses simple space-split pre-tokenization (no regex).
"""

import os
import json
import string
from functools import reduce
from collections import defaultdict, Counter


# =============================================================================
# TRAINING
# =============================================================================

def char_spaced_word_freq(raw_text):
    """Convert raw text into character-spaced word frequency dict with </w> boundary markers and handle any punctuation as an indvidual word"""
    raw_text = reduce(lambda text, punc: text.replace(punc, f" {punc} "), string.punctuation, raw_text)
    words = []
    s = ''
    for char in raw_text:
        if char == ' ':
            s += '</w>'
            words.append(s)
            s = ''
        else:
            s += char + ' '
    s += '</w>'
    words.append(s)
    freq = Counter(words)
    return dict(freq)


def bigram_pair_freq(char_spaced_freq):
    """Count bigram pair frequencies, weighted by word frequency."""
    bigram_freq_count = defaultdict(int)
    for word, freq in char_spaced_freq.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            bigram_freq_count[symbols[i], symbols[i + 1]] += freq
    return dict(bigram_freq_count)


def merge(bigram_pair, char_spaced_freq):
    """Merge the given bigram pair across the entire vocabulary."""
    pair = ' '.join(bigram_pair)
    replacement = ''.join(bigram_pair)
    merged_char_spaced_freq = defaultdict(int)
    for word, freq in char_spaced_freq.items():
        if pair in word:
            new_word = word.replace(pair, replacement)
            merged_char_spaced_freq[new_word] = freq
        else:
            merged_char_spaced_freq[word] = freq
    return dict(merged_char_spaced_freq)


def save_merge_rules(merge_rules, path="merge_rules.txt"):
    """Save merge rules to disk, one pair per line, in merge order."""
    if os.path.exists(path):
        with open(path, "a+") as f:
            for merge_pair in merge_rules:
                f.write(f"{merge_pair[0]} {merge_pair[1]}\n")
    else:
        with open(path, "w") as f:
            f.write("#version: 0.1\n")
            for merge_pair in merge_rules:
                f.write(f"{merge_pair[0]} {merge_pair[1]}\n")


# =============================================================================
# VOCAB
# =============================================================================
# Vocab consists of:
#   - Special tokens         (lowest IDs, fixed positions)
#   - Base chars from input  (sorted, deterministic ordering)
#   - </w> word boundary token
#   - All merged tokens      (one new token per merge rule)
#
# Uses </w> as word boundary marker instead of whitespace, as in the original paper.

def build_vocab(input_text, merge_rules, special_tokens=['[UNK]', '[PAD]', '[BOS]', '[EOS]']):
    """Build token->id vocabulary from base chars, special tokens, and merge results."""
    input_chars = sorted(list(set(input_text) - {' '}))
    input_chars.append('</w>')
    vocab = {token: i for i, token in enumerate(
        special_tokens + input_chars + list(map(''.join, merge_rules))
    )}
    return vocab


def save_vocab(vocab, path="vocab.json"):
    """Save vocab to disk as JSON."""
    with open(path, "w") as f:
        f.write(json.dumps(vocab))


# =============================================================================
# ENCODE
# =============================================================================
# encode() converts raw text to token IDs:
#   1. Pre-process: split into words, convert each to char-spaced format with </w>
#      e.g. "This word" -> ['T h i s </w>', 'w o r d </w>']
#   2. Replay merge rules (in order) across each word
#   3. Flatten into token list, look up each token's ID in vocab
#      Unknown tokens fall back to [UNK] ID.

def __preprocess(raw_text):
    """Convert raw text into list of char-spaced strings with </w> word boundaries."""
    words = raw_text.split(' ')
    preprocessed = []
    for word in words:
        tmp = []
        for char in word:
            tmp.append(char)
        tmp.append('</w>')
        preprocessed.append(tmp)
    preprocessed = list(map(' '.join, preprocessed))
    return preprocessed


def encode(input_text, vocab, merge_rules):
    """Encode raw text into (tokens, token_ids) by replaying BPE merge rules."""
    preprocessed_text = __preprocess(input_text)
    for pattern in merge_rules:
        replacement = ''.join(pattern.split(' '))
        for i, entry in enumerate(preprocessed_text):
            preprocessed_text[i] = entry.replace(pattern, replacement)
    tokens = [token for processed_word in preprocessed_text for token in processed_word.split(' ')]
    token_ids = [vocab.get(t, vocab['[UNK]']) for t in tokens]
    return tokens, token_ids


# =============================================================================
# DECODE
# =============================================================================
# decode() converts token IDs back to text:
#   1. Reverse lookup: each ID -> token string
#   2. Join all tokens, split on </w> to reconstruct words

def decode(token_ids, reverse_vocab):
    """Decode token IDs back to (tokens, words) via reverse vocab lookup."""
    tokens = [reverse_vocab.get(id, reverse_vocab[0]) for id in token_ids]
    words = list(map(str.strip, (''.join(tokens)).split("</w>")))
    return tokens, words

def train(n_merges, input_text):
    # --- Training ---
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    merge_rules = []
    char_spaced_freq = char_spaced_word_freq(input_text)
    print(f"Input: {input_text}")
    print(f"Initial Char Spaced Freq Dict: {char_spaced_freq}")

    for i in range(n_merges):
        print("*" * 50)
        print(f"Iteration {i}:")
        bigram_pairs = bigram_pair_freq(char_spaced_freq)
        print(f"Bigram Pairs Freq Dict: {bigram_pairs}")
        max_freq_bigram = max(bigram_pairs, key=bigram_pairs.get)
        print(f"Most frequent bigram pair: {max_freq_bigram}")
        print(f"Before merge: {char_spaced_freq}")
        char_spaced_freq = merge(max_freq_bigram, char_spaced_freq)
        print(f"After merge: {char_spaced_freq}")
        merge_rules.append(max_freq_bigram)
        print(f"Merge rules so far: {merge_rules}")

    vocab = build_vocab(input_text, merge_rules)
    save_merge_rules(merge_rules)
    save_vocab(vocab)

    print("-" * 60)
    print(f"Final Merge Rules: {merge_rules}")
    print(f"Vocab: {vocab}")
    print("-" * 60)

# =============================================================================
# MAIN — Training + encode/decode demo
# =============================================================================

if __name__ == "__main__":
    
    N_MERGES = 10
    INPUT_TEXT = "This text is raw text. don't stop, it's U.S.A. territory"

    train(N_MERGES, INPUT_TEXT)

    # --- Load artifacts ---
    with open('merge_rules.txt', 'r') as f:
        merge_rules_loaded = [line.strip() for line in f.readlines()]
        merge_rules_loaded.pop(0)  # remove version header

    with open('vocab.json', 'r') as f:
        vocab_loaded = json.loads(f.read())
        reverse_vocab = {v: k for k, v in vocab_loaded.items()}

    # --- Encode ---
    print("\n" + "=" * 60)
    print("ENCODE")
    print("=" * 60)
    tokens, token_ids = encode(INPUT_TEXT, vocab_loaded, merge_rules_loaded)
    print(f"Tokens:    {tokens}")
    print(f"Token IDs: {token_ids}")

    # --- Decode ---
    print("\n" + "=" * 60)
    print("DECODE")
    print("=" * 60)
    tokens_d, words = decode(token_ids, reverse_vocab)
    print(f"Tokens: {tokens_d}")
    print(f"Words:  {words}")

    # --- Roundtrip check ---
    print("\n" + "=" * 60)
    print("ROUNDTRIP CHECK")
    print("=" * 60)
    print(f"encode tokens == decode tokens: {tokens == tokens_d}")
    print(f"Reconstructed: '{' '.join(w for w in words if w)}'")
    print(f"Original:      '{INPUT_TEXT}'")