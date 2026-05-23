"""Text-level feature extraction for earnings call transcripts.

Extracts readability, specificity, and forward-looking statement features
based on academic research (Li 2008, Davis et al. 2012).
"""

import re
import math
from typing import Dict, List


# ── Forward-Looking Statement keywords ──
FLS_KEYWORDS = {
    "will", "expect", "expects", "expected", "expecting", "anticipate",
    "anticipates", "anticipated", "anticipating", "guidance", "outlook",
    "forecast", "project", "projected", "projecting", "plan", "plans",
    "planned", "planning", "believe", "believes", "intend", "intends",
    "estimate", "estimates", "estimated", "target", "targets", "goal",
    "goals", "aim", "aims", "predict", "predicts", "foresee", "foresees",
    "going forward", "next quarter", "next year", "fiscal year",
    "forward-looking", "future", "upcoming", "pipeline",
}

# ── Specificity patterns ──
NUMBER_PATTERN = re.compile(r'\b\d[\d,]*\.?\d*\b')
DOLLAR_PATTERN = re.compile(r'\$[\d,]+\.?\d*\s*(?:million|billion|thousand|M|B|K)?', re.IGNORECASE)
PERCENT_PATTERN = re.compile(r'\d+\.?\d*\s*%|percent', re.IGNORECASE)
BASIS_POINTS_PATTERN = re.compile(r'\d+\s*(?:basis points|bps)', re.IGNORECASE)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 10]


def _split_words(text: str) -> List[str]:
    """Split text into words."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def _count_syllables(word: str) -> int:
    """Estimate syllable count for a word."""
    word = word.lower().rstrip('e')
    vowels = 'aeiou'
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(1, count)


def readability_features(text: str) -> Dict[str, float]:
    """Compute readability metrics for a text.

    Returns:
        fog_index: Gunning Fog Index (higher = more complex)
        flesch_kincaid: Flesch-Kincaid Grade Level
        avg_sentence_length: Average words per sentence
        avg_word_length: Average characters per word
        vocab_richness: Unique words / total words (type-token ratio)
    """
    sentences = _split_sentences(text)
    words = _split_words(text)

    if not sentences or not words:
        return {
            "fog_index": 0.0,
            "flesch_kincaid": 0.0,
            "avg_sentence_length": 0.0,
            "avg_word_length": 0.0,
            "vocab_richness": 0.0,
        }

    n_sentences = len(sentences)
    n_words = len(words)
    avg_sent_len = n_words / n_sentences

    # Complex words (3+ syllables)
    complex_words = sum(1 for w in words if _count_syllables(w) >= 3)
    complex_pct = complex_words / n_words

    # Fog Index = 0.4 * (avg_sent_len + 100 * complex_pct)
    fog = 0.4 * (avg_sent_len + 100 * complex_pct)

    # Flesch-Kincaid Grade Level
    total_syllables = sum(_count_syllables(w) for w in words)
    avg_syllables = total_syllables / n_words
    fk = 0.39 * avg_sent_len + 11.8 * avg_syllables - 15.59

    # Average word length (characters)
    avg_word_len = sum(len(w) for w in words) / n_words

    # Vocabulary richness (type-token ratio)
    vocab_richness = len(set(words)) / n_words

    return {
        "fog_index": round(fog, 2),
        "flesch_kincaid": round(fk, 2),
        "avg_sentence_length": round(avg_sent_len, 2),
        "avg_word_length": round(avg_word_len, 2),
        "vocab_richness": round(vocab_richness, 4),
    }


def fls_features(text: str) -> Dict[str, float]:
    """Compute Forward-Looking Statement features.

    Returns:
        fls_ratio: Proportion of sentences containing FLS keywords
        fls_count: Total FLS keyword matches
    """
    sentences = _split_sentences(text)
    if not sentences:
        return {"fls_ratio": 0.0, "fls_count": 0}

    words_lower = set(_split_words(text))
    text_lower = text.lower()

    # Count FLS keywords found
    fls_matches = sum(1 for kw in FLS_KEYWORDS if kw in text_lower)

    # Count sentences with at least one FLS keyword
    fls_sentences = 0
    for sent in sentences:
        sent_lower = sent.lower()
        if any(kw in sent_lower for kw in FLS_KEYWORDS):
            fls_sentences += 1

    fls_ratio = fls_sentences / len(sentences)

    return {
        "fls_ratio": round(fls_ratio, 4),
        "fls_count": fls_matches,
    }


def specificity_features(text: str) -> Dict[str, float]:
    """Compute specificity/concreteness features.

    Returns:
        numbers_per_sentence: Average number mentions per sentence
        dollar_mentions: Count of dollar amount references
        percent_mentions: Count of percentage references
        specificity_score: Combined specificity metric
    """
    sentences = _split_sentences(text)
    n_sentences = max(len(sentences), 1)

    numbers = len(NUMBER_PATTERN.findall(text))
    dollars = len(DOLLAR_PATTERN.findall(text))
    percents = len(PERCENT_PATTERN.findall(text))
    bps = len(BASIS_POINTS_PATTERN.findall(text))

    # Combined score: weighted sum of specific references per sentence
    specificity = (numbers + dollars * 2 + percents * 2 + bps * 2) / n_sentences

    return {
        "numbers_per_sentence": round(numbers / n_sentences, 4),
        "dollar_mentions": dollars,
        "percent_mentions": percents + bps,
        "specificity_score": round(specificity, 4),
    }


def extract_all_text_features(text: str) -> Dict[str, float]:
    """Extract all text features from a single text block."""
    features = {}
    features.update(readability_features(text))
    features.update(fls_features(text))
    features.update(specificity_features(text))
    return features


def extract_call_features(chunks_texts: List[str]) -> Dict[str, float]:
    """Extract aggregated text features for an entire earnings call.

    Takes a list of chunk texts and returns call-level averages.
    """
    if not chunks_texts:
        return {}

    # Concatenate all text for call-level metrics
    full_text = " ".join(chunks_texts)
    call_features = extract_all_text_features(full_text)

    # Prefix with call_ to distinguish from chunk-level
    return {f"call_{k}": v for k, v in call_features.items()}
