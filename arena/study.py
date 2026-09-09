"""Run the detectors over the study sample (resumable)."""

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RESULTS_DIR

SAMPLE_CSV = RESULTS_DIR / "study_sample.csv"
SCORES_CSV = RESULTS_DIR / "scores.csv"
SUMMARY_JSON = RESULTS_DIR / "summary.json"
SCORE_COLUMNS = ["id", "detector", "support", "seconds", "error"]


def score_sample(sample: pd.DataFrame, detectors: list, scores_path=SCORES_CSV, log=print) -> None:
    """Score every (example, detector) pair not yet scored successfully; appends to scores.csv."""
    done = set()
    if scores_path.exists():
        prev = pd.read_csv(scores_path, dtype={"id": str})
        ok = prev[prev["support"].notna()]
        done = set(zip(ok["id"], ok["detector"]))
    new_file = not scores_path.exists()
    with open(scores_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(SCORE_COLUMNS)
        for det in detectors:
            todo = [r for r in sample.itertuples() if (str(r.id), det.key) not in done]
            log(f"[{det.key}] {len(todo)} examples to score")
            start = time.time()
            for i, r in enumerate(todo, 1):
                v = det.score(r.context, r.question, r.answer)
                writer.writerow([r.id, det.key, "" if np.isnan(v.support) else round(v.support, 6),
                                 round(v.seconds, 4), v.error[:300]])
                f.flush()
                if i % 20 == 0 or i == len(todo):
                    log(f"[{det.key}] {i}/{len(todo)}  ({(time.time() - start) / i:.2f}s/example)")


def load_scores(paths=None) -> pd.DataFrame:
    paths = paths or sorted(RESULTS_DIR.glob("scores*.csv"))
    s = pd.concat([pd.read_csv(p, dtype={"id": str}) for p in paths], ignore_index=True)
    s["ok"] = s["support"].notna()
    return s.sort_values("ok").drop_duplicates(["id", "detector"], keep="last").drop(columns="ok")
