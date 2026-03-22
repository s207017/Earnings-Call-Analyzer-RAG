# Agent 3: Risk Detection Agent

## Objective
Build a production-quality risk signal detection system for earnings call transcripts using a 10-category financial risk taxonomy.

## Tasks

### 1. Improve `src/agents/risk_detection.py`
The current scaffold uses basic keyword matching. Upgrade to:
- **Expanded taxonomy**: More comprehensive keywords for all 10 risk categories
- **Context-aware detection**: Avoid false positives (e.g., "no risk", "mitigated the risk", "reduced risk" should not flag as risk)
- **Negation handling**: Detect negation words (no, not, never, without, reduced, mitigated, eliminated) within a window before risk keywords
- **Severity scoring**: Classify detected risks as low/medium/high based on surrounding language (intensifiers like "significant", "major", "severe" vs "minor", "slight", "manageable")
- **Risk heatmap data**: Method to produce a companies × risk categories matrix for visualization
- **Temporal trends**: Track how risk signals change across quarters per company
- **Supervised classifier skeleton**: Add a class `SupervisedRiskClassifier` with training loop using transformers (fine-tune a BERT model for multi-label risk classification). This can be a skeleton/template that's ready to train when labeled data is available.

### 2. Create `configs/risk_taxonomy.json`
Full expanded taxonomy with:
- 10 risk categories
- 20-30 keywords/phrases per category
- Severity modifiers (intensifiers and diminishers)
- Negation words list

### 3. Create `src/utils/risk_utils.py`
Helper functions:
- Risk heatmap data preparation (for plotly)
- Risk trend formatting (time series per category)
- Risk summary generation (top risks per call)
- Color mapping for risk severity levels

## Files to Create/Modify
- `src/agents/risk_detection.py` (modify existing)
- `configs/risk_taxonomy.json` (new)
- `src/utils/risk_utils.py` (new)

## Dependencies
- `pandas`, `numpy`
- `transformers` (for supervised classifier skeleton)
- `torch`

## Quality Requirements
- Negation window should be configurable (default: 3 words before keyword)
- Severity scoring should be transparent and explainable
- False positive rate should be noticeably lower than naive keyword matching
- Code should be importable and usable by the Orchestrator agent
