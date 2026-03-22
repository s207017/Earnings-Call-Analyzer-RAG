"""Transcript parsing and chunking pipeline for earnings call transcripts."""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Speaker detection patterns (ordered by specificity)
# Name groups are limited to ~40 chars to avoid matching long phrases
# Names must have at least 2 words (first + last) or be a known role like "Operator"
SPEAKER_PATTERNS = [
    re.compile(r"^([A-Z][a-zA-Z\.\-']+(?:\s[A-Z][a-zA-Z\.\-']+){1,4})\s*--\s*(.+?)$", re.MULTILINE),      # Name -- Title
    re.compile(r"^([A-Z][a-zA-Z\.\-']+(?:\s[A-Z][a-zA-Z\.\-']+){1,4})\s*—\s*(.+?)$", re.MULTILINE),        # Name — Title (em dash)
    re.compile(r"^([A-Z][a-zA-Z\.\-']+(?:\s[A-Z][a-zA-Z\.\-']+){1,4})\s*\(([^)]+)\)\s*$", re.MULTILINE),   # Name (Title)\n at EOL
    re.compile(r"^([A-Z][a-zA-Z\.\-']+(?:\s[A-Z][a-zA-Z\.\-']+){1,4}),\s*(.+?):\s*$", re.MULTILINE),       # Name, Title:\n at EOL
    re.compile(r"^\[([A-Z][a-zA-Z\s\.\-']{1,40}?)\]\s*$", re.MULTILINE),                                     # [Name]
    re.compile(r"^([A-Z][a-zA-Z\.\-']+(?:\s[A-Z][a-zA-Z\.\-']+){1,4}):\s*$", re.MULTILINE),                 # Name:\n at EOL
    # Inline variants (speech continues on the same line after the colon)
    re.compile(r"^([A-Z][a-zA-Z\.\-']+(?:\s[A-Z][a-zA-Z\.\-']+)+),\s+(.+?):\s+(?=[A-Z])", re.MULTILINE),   # Name, Title: Speech...
    re.compile(r"^([A-Z][a-zA-Z\.\-']+(?:\s[A-Z][a-zA-Z\.\-']+)+)\s+\(([^)]+)\):\s+(?=[A-Z])", re.MULTILINE),  # Name (Company): Speech...
    re.compile(r"^([A-Z]{2,}(?:\s[A-Z]{2,})+):\s+", re.MULTILINE),                                           # ALL CAPS NAME: Speech...
    re.compile(r"^(Operator):\s*$", re.MULTILINE),                                                             # Operator:\n (standalone)
    re.compile(r"^(Operator):\s+", re.MULTILINE),                                                              # Operator: Speech...
]

# Words that should not be treated as speaker names
SPEAKER_BLACKLIST = {
    "prepared remarks", "question-and-answer session", "questions and answers",
    "forward-looking statements", "safe harbor",
}

# Section detection cues
PREPARED_REMARKS_CUES = [
    "prepared remarks", "opening remarks", "introductory remarks",
    "i'd now like to turn the call over", "i would like to turn the call over",
    "let me start with", "i'll begin with", "let me begin",
    "good morning", "good afternoon", "good evening",
    "welcome to", "thank you for joining",
]

QA_CUES = [
    "questions and answers:", "question-and-answer", "q&a session",
    "question and answer session",
    "we'll now take questions", "we will now take questions",
    "open the call to questions", "open it up for questions",
    "open the line for questions", "operator, please open",
    "first question", "our first question",
    "go to q&a", "move to q&a", "move over to q&a",
    "let's go to q&a", "let's move to q&a",
]

# Boilerplate patterns to remove (limited to ~500 chars to avoid consuming entire transcript)
BOILERPLATE_PATTERNS = [
    re.compile(r"(?i)safe\s+harbor.{0,500}?(?=\n\n|\n[A-Z][a-z])", re.DOTALL),
    re.compile(r"(?i)forward[- ]looking\s+statements?.{0,500}?(?=\n\n|\n[A-Z][a-z])", re.DOTALL),
    re.compile(r"(?i)this\s+(?:call|transcript)\s+(?:is\s+)?(?:being\s+)?recorded.*?\n"),
    re.compile(r"(?i)copyright\s+©?.*?\n"),
    re.compile(r"(?i)all\s+rights\s+reserved.*?\n"),
    re.compile(r"(?i)(?:the\s+)?motley\s+fool.*?\n"),
    re.compile(r"(?i)fool\.com.*?\n"),
]

# Role normalization
ROLE_KEYWORDS = {
    "CEO": ["chief executive", "ceo", "president and ceo"],
    "CFO": ["chief financial", "cfo", "finance officer"],
    "COO": ["chief operating", "coo"],
    "CTO": ["chief technology", "cto"],
    "VP": ["vice president", "vp", "svp", "evp"],
    "IR": ["investor relations", "ir "],
    "Analyst": ["analyst", "research", "securities", "capital", "bank", "morgan",
                "goldman", "barclays", "citi", "jpmorgan", "ubs", "wells fargo",
                "deutsche", "credit suisse", "bernstein", "piper", "canaccord",
                "raymond james", "cowen", "needham", "oppenheimer", "baird",
                "evercore", "keybanc", "mizuho", "bmo", "rbc", "loop capital",
                "moffett", "moffettnathanson", "jefferies", "wolfe",
                "stifel", "truist", "wedbush", "macquarie", "scotiabank",
                "william blair", "susquehanna", "rosenblatt"],
    "Operator": ["operator", "conference call"],
}


def classify_role(title: str) -> str:
    """Classify speaker role from their title."""
    if not title:
        return "Unknown"
    title_lower = title.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return role
    return "Other"


def detect_section(text: str) -> str:
    """Detect if text is from prepared remarks or Q&A section."""
    text_lower = text.lower()
    if any(cue in text_lower for cue in QA_CUES):
        return "qa"
    return "prepared_remarks"


def remove_boilerplate(text: str) -> str:
    """Remove legal disclaimers, copyright notices, and other boilerplate."""
    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_speakers(text: str) -> List[Dict]:
    """Extract speaker turns from transcript text using all matching patterns."""
    # Collect all matches from all patterns, dedup by position
    all_matches = []
    for pattern in SPEAKER_PATTERNS:
        for match in pattern.finditer(text):
            speaker = match.group(1).strip()
            title = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else ""
            # Skip blacklisted "speaker" names (section headers, etc.)
            if speaker.lower() in SPEAKER_BLACKLIST:
                continue
            # Clean speaker name: remove leading lines that aren't part of the name
            # (e.g., "Thanks.\nLuca Maestri" -> "Luca Maestri")
            if "\n" in speaker:
                speaker = speaker.split("\n")[-1].strip()
            all_matches.append({
                "start": match.start(),
                "end": match.end(),
                "speaker": speaker,
                "title": title,
            })

    if len(all_matches) < 2:
        logger.debug("No speakers detected, treating as single speaker")
        return [{"speaker": "Unknown", "title": "", "role": "Unknown", "text": text}]

    # Sort by position, deduplicate overlapping matches (keep earliest/longest)
    all_matches.sort(key=lambda m: (m["start"], -m["end"]))
    deduped = []
    for m in all_matches:
        if not deduped or m["start"] >= deduped[-1]["end"]:
            deduped.append(m)

    # Build speaker->role lookup from matches that have titles
    speaker_roles = {}
    for m in deduped:
        if m["title"]:
            role = classify_role(m["title"])
            if role not in ("Unknown", "Other"):
                speaker_roles[m["speaker"].lower()] = role

    # Scan intro text (first 20%) for role mentions near speaker names
    # Handles formats like "Satya Nadella, chief executive officer"
    intro_text = text[:len(text) // 5].lower()
    all_speaker_names = {m["speaker"].lower() for m in deduped}
    for speaker_lower in all_speaker_names:
        if speaker_lower in speaker_roles:
            continue  # Already have a role
        # Search all occurrences of this name in the intro
        search_start = 0
        while True:
            idx = intro_text.find(speaker_lower, search_start)
            if idx == -1:
                break
            # Look at the narrow context right after the name (up to next speaker or 80 chars)
            name_end = idx + len(speaker_lower)
            context = intro_text[name_end:name_end + 80]
            context = " ".join(context.split())  # Normalize whitespace (newlines -> spaces)
            # Truncate at the next speaker name to avoid bleeding into another person's title
            for other_name in all_speaker_names:
                if other_name != speaker_lower:
                    other_idx = context.find(other_name)
                    if other_idx != -1:
                        context = context[:other_idx]
            role = classify_role(context)
            if role not in ("Unknown", "Other"):
                speaker_roles[speaker_lower] = role
                break
            search_start = idx + 1

    turns = []
    for i, m in enumerate(deduped):
        # Try title first, then cached role from earlier appearance, then speaker name
        if m["title"]:
            role = classify_role(m["title"])
        elif m["speaker"].lower() in speaker_roles:
            role = speaker_roles[m["speaker"].lower()]
        else:
            role = classify_role(m["speaker"])
        start = m["end"]
        end = deduped[i + 1]["start"] if i + 1 < len(deduped) else len(text)
        speaker_text = text[start:end].strip()

        if speaker_text:
            turns.append({
                "speaker": m["speaker"],
                "title": m["title"],
                "role": role,
                "text": speaker_text,
            })

    # Post-pass: infer roles from speech content for unclassified speakers.
    # If a speaker's text mentions "investor relations", or they introduce
    # C-suite execs ("on the call with me", "joining us today"), they are IR.
    ir_cues = ["investor relations", "on the call with me", "joining us today are",
               "with me today are", "on today's call are"]
    for turn in turns:
        if turn["role"] in ("Unknown", "Other"):
            first_text = turn["text"][:500].lower()
            if any(cue in first_text for cue in ir_cues):
                speaker_roles[turn["speaker"].lower()] = "IR"
    # Apply inferred roles
    for turn in turns:
        if turn["role"] in ("Unknown", "Other"):
            inferred = speaker_roles.get(turn["speaker"].lower())
            if inferred:
                turn["role"] = inferred

    # ML fallback: use trained classifier for remaining Unknown/Other speakers
    unknown_turns = [t for t in turns if t["role"] in ("Unknown", "Other")]
    if unknown_turns:
        try:
            from src.agents.role_classifier import RoleClassifier
            clf = RoleClassifier()
            if clf.load():
                for turn in unknown_turns:
                    ml_role = clf.predict(turn["text"])
                    if ml_role != "Unknown":
                        # Cache so all turns from same speaker get same role
                        speaker_roles[turn["speaker"].lower()] = ml_role
                # Apply ML-inferred roles
                for turn in turns:
                    if turn["role"] in ("Unknown", "Other"):
                        inferred = speaker_roles.get(turn["speaker"].lower())
                        if inferred:
                            turn["role"] = inferred
        except Exception as e:
            logger.debug(f"ML role classifier not available: {e}")

    return turns


def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 50,
               min_chunk_size: int = 50) -> List[str]:
    """Split text into sentence-aware chunks of approximately chunk_size tokens."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        sent_len = len(words)

        if current_len + sent_len > chunk_size and current_chunk:
            chunk_text_str = " ".join(current_chunk)
            if len(chunk_text_str.split()) >= min_chunk_size:
                chunks.append(chunk_text_str)

            # Overlap: keep last few sentences
            overlap_words = 0
            overlap_start = len(current_chunk)
            for j in range(len(current_chunk) - 1, -1, -1):
                overlap_words += len(current_chunk[j].split())
                if overlap_words >= chunk_overlap:
                    overlap_start = j
                    break
            current_chunk = current_chunk[overlap_start:]
            current_len = sum(len(s.split()) for s in current_chunk)

        current_chunk.append(sentence)
        current_len += sent_len

    # Last chunk
    if current_chunk:
        chunk_text_str = " ".join(current_chunk)
        if len(chunk_text_str.split()) >= min_chunk_size:
            chunks.append(chunk_text_str)
        elif chunks:
            # Merge short last chunk with previous
            chunks[-1] = chunks[-1] + " " + chunk_text_str

    return chunks


class TranscriptParser:
    """Parses earnings call transcripts into structured, chunked data."""

    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = str(Path(__file__).parent.parent.parent)
        self.project_root = Path(project_root)

        config_path = self.project_root / "configs" / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.chunk_size = self.config["chunking"]["chunk_size"]
        self.chunk_overlap = self.config["chunking"]["chunk_overlap"]
        self.min_chunk_size = self.config["chunking"]["min_chunk_size"]

    def parse_transcript(self, text: str, ticker: str, quarter: str) -> List[Dict]:
        """Parse a single transcript into chunks with metadata."""
        # Clean
        text = remove_boilerplate(text)
        if not text or len(text) < 100:
            logger.warning(f"Transcript too short for {ticker} {quarter}")
            return []

        # Detect overall section boundaries
        # Skip early mentions (Operator often says "there will be a Q&A" in the intro)
        text_lower = text.lower()
        min_qa_pos = len(text) // 5  # Q&A never starts in the first 20%
        qa_start = len(text)
        for cue in QA_CUES:
            search_start = min_qa_pos
            while True:
                idx = text_lower.find(cue, search_start)
                if idx == -1 or idx >= qa_start:
                    break
                qa_start = idx
                break

        # Extract speakers
        turns = extract_speakers(text)
        chunks_out = []
        chunk_index = 0

        for turn in turns:
            # Determine section based on position
            turn_pos = text.find(turn["text"][:100]) if len(turn["text"]) >= 100 else 0
            section = "qa" if turn_pos >= qa_start else "prepared_remarks"

            # Chunk this speaker's text
            text_chunks = chunk_text(
                turn["text"], self.chunk_size, self.chunk_overlap, self.min_chunk_size
            )

            for chunk_text_str in text_chunks:
                chunks_out.append({
                    "chunk_id": f"{ticker}_{quarter}_{chunk_index:04d}",
                    "ticker": ticker,
                    "quarter": quarter,
                    "speaker": turn["speaker"],
                    "role": turn["role"],
                    "section": section,
                    "text": chunk_text_str,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

        logger.info(f"Parsed {ticker} {quarter}: {len(turns)} speaker turns -> {len(chunks_out)} chunks")
        return chunks_out

    def process_all(self, manifest_path: str = None) -> pd.DataFrame:
        """Process all transcripts from raw directory into chunks."""
        raw_dir = self.project_root / self.config["paths"]["raw"]
        if not raw_dir.exists():
            logger.error(f"Raw directory not found: {raw_dir}")
            return pd.DataFrame()

        all_chunks = []
        for ticker_dir in sorted(raw_dir.iterdir()):
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name
            for txt_file in sorted(ticker_dir.glob("*.txt")):
                quarter = txt_file.stem
                try:
                    text = txt_file.read_text(encoding="utf-8", errors="replace")
                    chunks = self.parse_transcript(text, ticker, quarter)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error processing {ticker}/{quarter}: {e}")

        if not all_chunks:
            logger.warning("No chunks produced!")
            return pd.DataFrame()

        df = pd.DataFrame(all_chunks)

        # Save
        out_path = self.project_root / self.config["paths"]["processed"] / "all_chunks.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(df)} chunks to {out_path}")
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = TranscriptParser()
    df = parser.process_all()
    print(f"Total chunks: {len(df)}")
    if not df.empty:
        print(f"Companies: {df['ticker'].nunique()}")
        print(f"Quarters: {df['quarter'].nunique()}")
        print(f"Sections: {df['section'].value_counts().to_dict()}")
        print(f"Roles: {df['role'].value_counts().to_dict()}")
