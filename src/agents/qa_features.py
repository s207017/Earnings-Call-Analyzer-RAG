"""Q&A dynamics and evasion detection features for earnings calls.

Extracts behavioral signals from the Q&A section based on research by
Hobson, Mayew & Venkatachalam (2012) and Brockman, Li & Price (2015).
"""

import re
from typing import Dict, List

import pandas as pd


# ── Evasion / deflection cues ──
DEFLECTION_PHRASES = [
    "as i mentioned", "as we mentioned", "as i said", "as we said",
    "as i noted", "as we noted", "as discussed",
    "i think i addressed", "we already covered",
    "let me come back to", "let me get back to",
    "i'll let", "i would refer you to",
    "we'll see", "we will see", "time will tell",
    "it's hard to say", "it's difficult to predict",
    "i don't want to speculate", "we don't speculate",
    "i'm not going to get into", "we're not going to comment",
    "i can't really", "we can't really",
    "that's a great question", "that's a good question",
    "it depends", "it really depends",
    "we'll have more to say", "more to come on that",
    "stay tuned", "too early to tell",
]

# Filler / hedge words
FILLER_WORDS = {
    "um", "uh", "you know", "sort of", "kind of", "basically",
    "actually", "honestly", "frankly", "obviously", "clearly",
    "essentially", "effectively", "literally",
}

# Passive voice indicators (simplified)
PASSIVE_PATTERN = re.compile(
    r'\b(?:is|are|was|were|been|be|being)\s+(?:\w+ly\s+)?'
    r'(?:\w+ed|driven|given|taken|made|done|seen|known|shown|found)\b',
    re.IGNORECASE,
)

# Pronoun patterns
FIRST_PERSON = re.compile(r'\b(?:I|we|our|us|my)\b', re.IGNORECASE)
THIRD_PERSON = re.compile(r'\b(?:it|they|them|the company|the team|the business|management)\b', re.IGNORECASE)


def _word_count(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))


def _sentence_count(text: str) -> int:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return max(len([s for s in sents if len(s) > 10]), 1)


def qa_dynamics_features(chunks_df: pd.DataFrame) -> Dict[str, float]:
    """Extract Q&A behavioral features from chunks dataframe.

    Expects columns: text, role, section
    """
    features = {}

    qa_chunks = chunks_df[chunks_df["section"] == "qa"] if "section" in chunks_df.columns else pd.DataFrame()
    prepared_chunks = chunks_df[chunks_df["section"] == "prepared_remarks"] if "section" in chunks_df.columns else pd.DataFrame()

    if qa_chunks.empty:
        return {
            "qa_mgmt_word_ratio": 0.0,
            "qa_avg_answer_length": 0.0,
            "qa_avg_question_length": 0.0,
            "qa_answer_question_ratio": 0.0,
            "qa_deflection_count": 0,
            "qa_deflection_rate": 0.0,
            "qa_filler_rate": 0.0,
            "qa_passive_rate": 0.0,
            "qa_pronoun_shift": 0.0,
            "qa_specificity_drop": 0.0,
        }

    # Management vs analyst word counts in Q&A
    mgmt_roles = {"CEO", "CFO", "COO", "CTO", "President", "Executive",
                   "Chief", "VP", "Director", "Officer", "Head"}
    analyst_role = "Analyst"

    if "role" in qa_chunks.columns:
        mgmt_qa = qa_chunks[qa_chunks["role"].isin(mgmt_roles) | qa_chunks["role"].str.contains("CEO|CFO|COO|CTO|VP|Director|President", case=False, na=False)]
        analyst_qa = qa_chunks[qa_chunks["role"] == analyst_role]

        mgmt_words = sum(_word_count(t) for t in mgmt_qa["text"])
        analyst_words = sum(_word_count(t) for t in analyst_qa["text"])

        features["qa_mgmt_word_ratio"] = mgmt_words / max(mgmt_words + analyst_words, 1)

        # Average answer vs question length
        mgmt_lengths = [_word_count(t) for t in mgmt_qa["text"]]
        analyst_lengths = [_word_count(t) for t in analyst_qa["text"]]

        features["qa_avg_answer_length"] = sum(mgmt_lengths) / max(len(mgmt_lengths), 1)
        features["qa_avg_question_length"] = sum(analyst_lengths) / max(len(analyst_lengths), 1)
        features["qa_answer_question_ratio"] = (
            features["qa_avg_answer_length"] / max(features["qa_avg_question_length"], 1)
        )
    else:
        features["qa_mgmt_word_ratio"] = 0.0
        features["qa_avg_answer_length"] = 0.0
        features["qa_avg_question_length"] = 0.0
        features["qa_answer_question_ratio"] = 0.0

    # Deflection detection in management Q&A answers
    mgmt_texts = mgmt_qa["text"].tolist() if "role" in qa_chunks.columns else qa_chunks["text"].tolist()
    n_mgmt_chunks = max(len(mgmt_texts), 1)

    deflection_count = 0
    for text in mgmt_texts:
        text_lower = text.lower()
        for phrase in DEFLECTION_PHRASES:
            if phrase in text_lower:
                deflection_count += 1
                break  # count max once per chunk

    features["qa_deflection_count"] = deflection_count
    features["qa_deflection_rate"] = round(deflection_count / n_mgmt_chunks, 4)

    # Filler word rate
    all_mgmt_text = " ".join(mgmt_texts).lower()
    total_words = max(_word_count(all_mgmt_text), 1)
    filler_count = sum(all_mgmt_text.count(f) for f in FILLER_WORDS)
    features["qa_filler_rate"] = round(filler_count / total_words, 4)

    # Passive voice rate
    passive_count = len(PASSIVE_PATTERN.findall(all_mgmt_text))
    n_sentences = _sentence_count(all_mgmt_text)
    features["qa_passive_rate"] = round(passive_count / n_sentences, 4)

    # Pronoun shift: compare first-person vs third-person ratio
    # between prepared remarks and Q&A (shift to third person = distancing)
    if not prepared_chunks.empty and "role" in prepared_chunks.columns:
        prep_mgmt = prepared_chunks[prepared_chunks["role"].isin(mgmt_roles) | prepared_chunks["role"].str.contains("CEO|CFO|COO|CTO|VP|Director|President", case=False, na=False)]
        if not prep_mgmt.empty:
            prep_text = " ".join(prep_mgmt["text"].tolist()).lower()
            prep_first = len(FIRST_PERSON.findall(prep_text))
            prep_third = len(THIRD_PERSON.findall(prep_text))
            prep_ratio = prep_first / max(prep_third, 1)

            qa_first = len(FIRST_PERSON.findall(all_mgmt_text))
            qa_third = len(THIRD_PERSON.findall(all_mgmt_text))
            qa_ratio = qa_first / max(qa_third, 1)

            # Positive = more first-person in prepared vs Q&A (distancing in Q&A)
            features["qa_pronoun_shift"] = round(prep_ratio - qa_ratio, 4)
        else:
            features["qa_pronoun_shift"] = 0.0
    else:
        features["qa_pronoun_shift"] = 0.0

    # Specificity drop: compare number density in prepared vs Q&A
    from src.agents.text_features import NUMBER_PATTERN
    if not prepared_chunks.empty:
        prep_all = " ".join(prepared_chunks["text"].tolist())
        prep_nums = len(NUMBER_PATTERN.findall(prep_all)) / max(_sentence_count(prep_all), 1)
        qa_all = " ".join(qa_chunks["text"].tolist())
        qa_nums = len(NUMBER_PATTERN.findall(qa_all)) / max(_sentence_count(qa_all), 1)
        features["qa_specificity_drop"] = round(prep_nums - qa_nums, 4)
    else:
        features["qa_specificity_drop"] = 0.0

    return features
