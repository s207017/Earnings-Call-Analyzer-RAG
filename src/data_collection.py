"""Data collection module for loading earnings call transcripts from Kaggle datasets."""

import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Mapping from AlphaSpread folder names to tickers
FOLDER_TO_TICKER = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Alphabet": "GOOGL", "Amazon": "AMZN",
    "Meta": "META", "META": "META", "Nvidia": "NVDA", "NVIDIA": "NVDA",
    "Tesla": "TSLA", "Netflix": "NFLX", "Salesforce": "CRM", "SalesForce": "CRM",
    "Oracle": "ORCL", "Google": "GOOGL", "Facebook": "META",
    "AMD": "AMD", "Cisco": "CSCO", "IBM": "IBM", "Adobe": "ADBE",
    "Accenture": "ACN", "PAYPAL": "PYPL", "Mastercard": "MA",
    "AXP": "AXP", "JPM": "JPM", "Bank of America": "BAC", "Citi": "C",
    "Deutsche Bank": "DB", "Costco": "COST", "Walmart": "WMT",
    "Nike": "NKE", "Lululemon": "LULU", "Walt Disney": "DIS",
    "Marriott": "MAR", "BKNG": "BKNG", "Ford": "F", "GM": "GM",
    "Allstate": "ALL", "Cardinal Health": "CAH", "Elevance Health": "ELV",
    "UnitedHealth": "UNH", "BP": "BP", "Shell": "SHEL",
    "BAM": "BAM", "Volvo": "VOLVY", "BMW": "BMW.DE", "SIE": "SIE.DE",
    "BBVA": "BBVA", "Airbus": "AIR.PA", "Amadeus": "AMS.PA",
    "Capgemini": "CAP.PA", "Loreal": "OR.PA", "Louis Vuitton": "MC.PA",
    "Schneider Electric": "SCHNEIDER.PA", "Engie": "ENGI.PA",
    "EDF": "EDF.PA", "Compass Group": "CPG.L", "Branco": "BBAS3.SA",
}

TICKER_ALIASES = {
    "GOOG": "GOOGL", "FB": "META", "AMZN.COM": "AMZN",
    "ANTM": "ELV",  # Anthem renamed to Elevance
}


class DataCollector:
    """Collects and normalizes earnings call transcripts from multiple Kaggle datasets."""

    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = str(Path(__file__).parent.parent)
        self.project_root = Path(project_root)

        config_path = self.project_root / "configs" / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        companies_path = self.project_root / "configs" / "companies.json"
        with open(companies_path) as f:
            self.companies = json.load(f)

        self.target_tickers = {c["ticker"] for c in self.companies}
        self.target_quarters = set(self.config["quarters"])
        self.raw_dir = self.project_root / self.config["paths"]["raw"]
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker symbol to standard form."""
        t = ticker.strip().upper()
        return TICKER_ALIASES.get(t, t)

    def _normalize_quarter(self, quarter_str: str) -> str:
        """Normalize quarter string to format like 2023Q1."""
        q = quarter_str.strip().upper().replace(" ", "").replace("-", "")
        # Handle formats: "Q1 2023", "2023-Q1", "2023Q1", "FY2023Q1"
        m = re.match(r"(?:FY)?(\d{4})Q(\d)", q)
        if m:
            return f"{m.group(1)}Q{m.group(2)}"
        m = re.match(r"Q(\d)(\d{4})", q)
        if m:
            return f"{m.group(2)}Q{m.group(1)}"
        return q

    def load_motley_fool(self) -> pd.DataFrame:
        """Load transcripts from Motley Fool pickle dataset."""
        pkl_path = self.project_root / self.config["paths"]["kaggle_motley_fool"]
        if not pkl_path.exists():
            logger.warning(f"Motley Fool pickle not found at {pkl_path}")
            return pd.DataFrame()

        logger.info(f"Loading Motley Fool data from {pkl_path}")
        df = pd.read_pickle(pkl_path)
        logger.info(f"Loaded {len(df)} transcripts from Motley Fool")

        # Normalize columns
        df.columns = [c.lower().strip() for c in df.columns]
        records = []
        for _, row in df.iterrows():
            ticker = self._normalize_ticker(str(row.get("ticker", "")))
            if ticker not in self.target_tickers:
                continue
            quarter = self._normalize_quarter(str(row.get("q", "")))
            if quarter not in self.target_quarters:
                continue
            transcript = str(row.get("transcript", ""))
            if len(transcript) < 100:
                continue
            records.append({
                "ticker": ticker,
                "quarter": quarter,
                "date": str(row.get("date", "")),
                "transcript_text": transcript,
                "source": "motley_fool",
            })

        result = pd.DataFrame(records)
        logger.info(f"Filtered to {len(result)} Motley Fool transcripts for target companies/quarters")
        return result

    def _map_folder_to_ticker(self, folder_name: str) -> str:
        """Map AlphaSpread folder name to ticker."""
        if folder_name in FOLDER_TO_TICKER:
            return FOLDER_TO_TICKER[folder_name]
        # Try case-insensitive match
        for key, val in FOLDER_TO_TICKER.items():
            if key.lower() == folder_name.lower():
                return val
        return folder_name.upper()

    def _parse_alphaspread_filename(self, filename: str) -> dict:
        """Parse quarter and ticker from AlphaSpread filename like 2023_Q3_aapl_processed.txt."""
        name = Path(filename).stem
        # Patterns: 2023_Q3_aapl_processed, 2023_Q3_aapl, 2024_Q2_aapl
        m = re.match(r"(\d{4})_Q(\d)_(\w+?)(?:_processed)?$", name, re.IGNORECASE)
        if m:
            year, q, ticker = m.group(1), m.group(2), m.group(3)
            return {"quarter": f"{year}Q{q}", "ticker": self._normalize_ticker(ticker)}
        return None

    def load_alphaspread(self) -> pd.DataFrame:
        """Load transcripts from AlphaSpread dataset (both cleaned and NLP dirs)."""
        records = []
        seen = set()

        for path_key in ["kaggle_alphaspread_cleaned", "kaggle_alphaspread_nlp"]:
            base_dir = self.project_root / self.config["paths"][path_key]
            if not base_dir.exists():
                logger.warning(f"AlphaSpread directory not found: {base_dir}")
                continue

            logger.info(f"Scanning AlphaSpread directory: {base_dir}")
            for company_dir in sorted(base_dir.iterdir()):
                if not company_dir.is_dir() or company_dir.name.startswith("."):
                    continue
                ticker = self._map_folder_to_ticker(company_dir.name)
                if ticker not in self.target_tickers:
                    continue

                for txt_file in sorted(company_dir.glob("*.txt")):
                    parsed = self._parse_alphaspread_filename(txt_file.name)
                    if parsed is None:
                        continue
                    quarter = parsed["quarter"]
                    if quarter not in self.target_quarters:
                        continue

                    key = (ticker, quarter, path_key)
                    if key in seen:
                        continue
                    seen.add(key)

                    try:
                        text = txt_file.read_text(encoding="utf-8", errors="replace")
                    except Exception as e:
                        logger.error(f"Error reading {txt_file}: {e}")
                        continue

                    if len(text) < 100:
                        continue

                    records.append({
                        "ticker": ticker,
                        "quarter": quarter,
                        "date": "",
                        "transcript_text": text,
                        "source": "alphaspread",
                    })

        result = pd.DataFrame(records)
        logger.info(f"Loaded {len(result)} AlphaSpread transcripts for target companies/quarters")
        return result

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep the longer transcript when both datasets have the same ticker+quarter."""
        df["text_len"] = df["transcript_text"].str.len()
        df = df.sort_values("text_len", ascending=False).drop_duplicates(
            subset=["ticker", "quarter"], keep="first"
        )
        df = df.drop(columns=["text_len"])
        logger.info(f"After deduplication: {len(df)} transcripts")
        return df.reset_index(drop=True)

    def save_raw(self, df: pd.DataFrame):
        """Save individual transcript files to data/raw/{ticker}/{quarter}.txt."""
        for _, row in df.iterrows():
            ticker_dir = self.raw_dir / row["ticker"]
            ticker_dir.mkdir(parents=True, exist_ok=True)
            out_path = ticker_dir / f"{row['quarter']}.txt"
            out_path.write_text(row["transcript_text"], encoding="utf-8")
        logger.info(f"Saved {len(df)} transcripts to {self.raw_dir}")

    def save_manifest(self, df: pd.DataFrame):
        """Create data/manifest.json listing all transcripts."""
        manifest = []
        for _, row in df.iterrows():
            manifest.append({
                "file_path": f"data/raw/{row['ticker']}/{row['quarter']}.txt",
                "ticker": row["ticker"],
                "quarter": row["quarter"],
                "date": row["date"],
                "source": row["source"],
            })
        manifest_path = self.project_root / "data" / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest saved to {manifest_path}")

    def coverage_report(self, df: pd.DataFrame):
        """Print coverage report showing available vs missing company-quarters."""
        available = set(zip(df["ticker"], df["quarter"]))
        print("\n=== Coverage Report ===")
        print(f"{'Ticker':<8}", end="")
        for q in sorted(self.target_quarters):
            print(f"{q:<10}", end="")
        print()
        for ticker in sorted(self.target_tickers):
            print(f"{ticker:<8}", end="")
            for q in sorted(self.target_quarters):
                status = "✓" if (ticker, q) in available else "✗"
                print(f"{status:<10}", end="")
            print()
        total = len(self.target_tickers) * len(self.target_quarters)
        found = len(available)
        print(f"\nCoverage: {found}/{total} ({found/total*100:.0f}%)\n")

    def collect(self) -> pd.DataFrame:
        """Run the full collection pipeline."""
        logger.info("Starting data collection...")
        mf_df = self.load_motley_fool()
        as_df = self.load_alphaspread()

        if mf_df.empty and as_df.empty:
            logger.error("No transcripts found from any source!")
            return pd.DataFrame()

        combined = pd.concat([as_df, mf_df], ignore_index=True)
        logger.info(f"Combined: {len(combined)} transcripts before dedup")

        deduped = self.deduplicate(combined)
        self.save_raw(deduped)
        self.save_manifest(deduped)
        self.coverage_report(deduped)
        return deduped


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    collector = DataCollector()
    df = collector.collect()
    print(f"Collected {len(df)} transcripts")
