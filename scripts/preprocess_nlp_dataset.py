"""
Preprocess Kaggle NLP_Dataset into the pipeline's expected format.

Reads data/kaggle/archive (2)/NLP_Dataset/NLP_Dataset/{Company}/*.txt
and writes to data/raw/{TICKER}/{QUARTER}.txt

The NLP_Dataset files have a different format (tab-separated header + body).
This script cleans them into plain-text transcripts matching the Motley Fool format.

Usage:
    python3.11 scripts/preprocess_nlp_dataset.py [--dry-run]
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
NLP_DIR = PROJECT_ROOT / "data" / "kaggle" / "archive (2)" / "NLP_Dataset" / "NLP_Dataset"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"

# Map company folder names to tickers
COMPANY_TO_TICKER = {
    "AMD": "AMD", "AXP": "AXP", "Accenture": "ACN", "Adobe": "ADBE",
    "Airbus": "AIR.PA", "Allstate": "ALL", "Alphabet": "GOOGL", "Amadeus": "AMS.PA",
    "Amazon": "AMZN", "Apple": "AAPL", "BAM": "BAM", "BBVA": "BBVA",
    "BKNG": "BKNG", "BMW": "BMW.DE", "BP": "BP", "Bank of America": "BAC",
    "Capgemini": "CAP.PA", "Cardinal Health": "CAH", "Cisco": "CSCO",
    "Citi": "C", "Costco": "COST", "Deutsche Bank": "DB",
    "EDF": "EDF.PA", "Elevance Health": "ELV", "Engie": "ENGI.PA",
    "Ford": "F", "GM": "GM", "IBM": "IBM", "JPM": "JPM",
    "Loreal": "OR.PA", "Louis Vuitton": "MC.PA", "Lululemon": "LULU",
    "META": "META", "Marriott": "MAR", "Mastercard": "MA",
    "Microsoft": "MSFT", "Nike": "NKE", "Nvidia": "NVDA", "Oracle": "ORCL",
    "PAYPAL": "PYPL", "SIE": "SIE.DE", "SalesForce": "CRM",
    "Schneider Electric": "SCHNEIDER.PA", "Shell": "SHEL",
    "UnitedHealth": "UNH", "Walmart": "WMT", "Walt Disney": "DIS",
    "Volvo": "VOLVY",
}


def parse_filename(filename: str) -> tuple:
    """Extract quarter from filename like '2022_Q1_aapl.txt' → '2022Q1'."""
    m = re.match(r"(\d{4})_Q(\d)_", filename)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    return None


def clean_nlp_transcript(raw_text: str) -> str:
    """Clean the NLP_Dataset format into readable transcript text.

    The NLP_Dataset files have a messy header line followed by tab-separated
    speaker/text blocks. This converts them to a clean format.
    """
    lines = raw_text.split("\n")
    if not lines:
        return ""

    # First line is usually a metadata header — skip it if it contains
    # "Earnings Call Transcript" with no tab-separated content we want
    header = lines[0]
    body_lines = lines[1:] if "Earnings Call Transcript" in header else lines

    cleaned = []
    for line in body_lines:
        # Lines are often tab-separated: index\tSpeaker\tText
        parts = line.split("\t")
        if len(parts) >= 3:
            # Format: [index, speaker, text]
            try:
                int(parts[0])  # first field is numeric index
                speaker = parts[1].strip()
                text = parts[2].strip()
                if speaker and text:
                    cleaned.append(f"{speaker}\n{text}\n")
                elif text:
                    cleaned.append(f"{text}\n")
            except ValueError:
                # Not an indexed line, join everything
                cleaned.append(line.strip() + "\n")
        elif len(parts) == 2:
            cleaned.append(f"{parts[0].strip()}\n{parts[1].strip()}\n")
        elif line.strip():
            cleaned.append(line.strip() + "\n")

    result = "\n".join(cleaned)

    # If cleaning produced very little, fall back to raw text
    if len(result) < len(raw_text) * 0.3:
        return raw_text

    return result


def load_manifest() -> list:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return []


def save_manifest(entries: list):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Preprocess NLP_Dataset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not NLP_DIR.exists():
        print(f"ERROR: {NLP_DIR} not found")
        sys.exit(1)

    manifest = load_manifest()
    existing = {(e["ticker"], e["quarter"]) for e in manifest}

    written = 0
    skipped_exists = 0
    skipped_short = 0
    skipped_unknown = 0
    new_entries = []

    company_dirs = sorted([d for d in NLP_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(company_dirs)} company folders")

    for company_dir in company_dirs:
        company_name = company_dir.name
        ticker = COMPANY_TO_TICKER.get(company_name)

        if not ticker:
            # Try to find in the folder — some have odd names
            print(f"  WARNING: No ticker mapping for '{company_name}', skipping")
            skipped_unknown += 1
            continue

        txt_files = sorted(company_dir.glob("*.txt"))
        for txt_file in txt_files:
            quarter = parse_filename(txt_file.name)
            if not quarter:
                continue

            if (ticker, quarter) in existing:
                skipped_exists += 1
                continue

            raw_text = txt_file.read_text(encoding="utf-8", errors="replace")
            cleaned = clean_nlp_transcript(raw_text)

            if len(cleaned) < 500:
                skipped_short += 1
                continue

            out_path = RAW_DIR / ticker / f"{quarter}.txt"

            if out_path.exists():
                skipped_exists += 1
                existing.add((ticker, quarter))
                continue

            if args.dry_run:
                print(f"  [DRY RUN] {company_name}/{txt_file.name} → {out_path} ({len(cleaned):,} chars)")
                written += 1
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(cleaned, encoding="utf-8")

            new_entries.append({
                "file_path": str(out_path.relative_to(PROJECT_ROOT)),
                "ticker": ticker,
                "quarter": quarter,
                "date": "",
                "source": "nlp_dataset",
            })
            existing.add((ticker, quarter))
            written += 1

    if new_entries and not args.dry_run:
        manifest.extend(new_entries)
        save_manifest(manifest)

    print(f"\nResults:")
    print(f"  Written:  {written} new transcripts")
    print(f"  Skipped:  {skipped_exists} (already exist) + {skipped_short} (too short) + {skipped_unknown} (unknown company)")
    print(f"  Manifest: {len(manifest)} total entries")


if __name__ == "__main__":
    main()
