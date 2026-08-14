"""HaluBench loading and the study sample (balanced per source, dev/test split)."""

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DATA_DIR

DATASET_ID = "PatronusAI/HaluBench"  # CC-BY-NC-2.0, Patronus AI
PARQUET = "data/test-00000-of-00001.parquet"
SOURCE_NAMES = {
    "halueval": "HaluEval (general QA)",
    "DROP": "DROP (reading comprehension)",
    "covidQA": "CovidQA (biomedical)",
    "pubmedQA": "PubMedQA (biomedical)",
    "FinanceBench": "FinanceBench (finance)",
    "RAGTruth": "RAGTruth (open RAG answers)",
}
