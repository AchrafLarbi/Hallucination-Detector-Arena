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


def load_halubench(path: Optional[Path] = None) -> pd.DataFrame:
    """All 14,900 HaluBench rows with a binary `hallucinated` column (FAIL = 1)."""
    if path is None:
        path = DATA_DIR / PARQUET
        if not path.exists():
            from huggingface_hub import hf_hub_download

            hf_hub_download(DATASET_ID, PARQUET, repo_type="dataset", local_dir=DATA_DIR)
    df = pd.read_parquet(path)
    return df.rename(columns={"passage": "context", "source_ds": "source"}).assign(
        hallucinated=(df["label"] == "FAIL").astype(int),
        question=df["question"].fillna("").astype(str),
        answer=df["answer"].fillna("").astype(str),
    )
