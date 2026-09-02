"""Run the detector validity study on HaluBench."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from arena import config
from arena.data import load_halubench, sample_study


def make_detectors(keys):
    from arena import detectors as D

    dets = []
    for key in keys:
        dets.append({c.key: c for c in D.LOCAL_DETECTORS}[key]())
    return dets


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--per-label", type=int, default=40)
    s.add_argument("--max-context-words", type=int, default=1500)
    s.add_argument("--seed", type=int, default=13)
    sc = sub.add_parser("score")
    sc.add_argument("detectors", nargs="+", choices=["nli", "minicheck", "hhem"])
    args = p.parse_args()
    config.RESULTS_DIR.mkdir(exist_ok=True)

    if args.cmd == "sample":
        sample = sample_study(load_halubench(), args.per_label, args.max_context_words, args.seed)
        sample.to_csv(config.RESULTS_DIR / "study_sample.csv", index=False)
        print(sample.groupby(["source", "split", "hallucinated"]).size().unstack(fill_value=0))
        print(f"{len(sample)} examples -> {config.RESULTS_DIR / 'study_sample.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
