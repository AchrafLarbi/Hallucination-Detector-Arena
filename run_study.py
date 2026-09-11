"""Run the detector validity study on HaluBench.

    python run_study.py sample                      # draw the balanced sample -> results/study_sample.csv
    python run_study.py score nli minicheck hhem    # local detectors (CPU, ~minutes)
    python run_study.py score llm_judge             # LLM judge (needs LLM_API_KEY; paced for Groq's free tier)
    python run_study.py analyze                     # -> results/summary.json

Scoring is resumable: re-running only scores the missing (example, detector) pairs.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from arena import config
from arena.data import load_halubench, sample_study
from arena.study import SAMPLE_CSV, SCORES_CSV, SUMMARY_JSON, analyze, load_scores, report_markdown, score_sample


def make_detectors(keys):
    from arena import detectors as D

    dets = []
    for key in keys:
        if key == "llm_judge":
            from arena.llm import LLMClient

            client = LLMClient(config.LLM_API_KEY, config.LLM_BASE_URL, config.LLM_JUDGE_MODEL, temperature=0.0,
                               max_queries_per_minute=config.LLM_MAX_QPM, max_tokens_per_day=config.LLM_MAX_TOKENS_PER_DAY,
                               max_retries=8, extra_body={"reasoning_effort": "low"})
            dets.append(D.LLMJudge(client))
        else:
            dets.append({c.key: c for c in D.LOCAL_DETECTORS}[key]())
    return dets


def detector_names() -> dict:
    from arena.detectors import LOCAL_DETECTORS

    names = {c.key: c.name for c in LOCAL_DETECTORS}
    names["llm_judge"] = f"LLM judge ({config.LLM_JUDGE_MODEL.split('/')[-1]})"
    return names


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--per-label", type=int, default=40)
    s.add_argument("--max-context-words", type=int, default=1500)
    s.add_argument("--seed", type=int, default=13)
    sc = sub.add_parser("score")
    sc.add_argument("detectors", nargs="+", choices=["nli", "minicheck", "hhem", "llm_judge"])
    sc.add_argument("--out", default=str(SCORES_CSV), help="scores file (run detectors in parallel into separate files)")
    sub.add_parser("analyze")
    args = p.parse_args()
    config.RESULTS_DIR.mkdir(exist_ok=True)

    if args.cmd == "sample":
        sample = sample_study(load_halubench(), args.per_label, args.max_context_words, args.seed)
        sample.to_csv(SAMPLE_CSV, index=False)
        print(sample.groupby(["source", "split", "hallucinated"]).size().unstack(fill_value=0))
        print(f"{len(sample)} examples -> {SAMPLE_CSV}")
    elif args.cmd == "score":
        sample = pd.read_csv(SAMPLE_CSV, dtype={"id": str}, keep_default_na=False)
        score_sample(sample, make_detectors(args.detectors), scores_path=Path(args.out), log=lambda m: print(m, flush=True))
    elif args.cmd == "analyze":
        sample = pd.read_csv(SAMPLE_CSV, dtype={"id": str}, keep_default_na=False)
        report = analyze(sample, load_scores(), detector_names())
        SUMMARY_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
        (SUMMARY_JSON.parent / "RESULTS.md").write_text(report_markdown(report), encoding="utf-8")
        for key, r in report["detectors"].items():
            print(f"{r['name']:38} AUROC {r['test_auroc']:.3f} {r['test_auroc_ci']}  "
                  f"BalAcc@0.5 {r['test_bal_acc_default']:.3f}  BalAcc@cal({r['calibrated_threshold']}) "
                  f"{r['test_bal_acc_calibrated']:.3f}  failed {r['n_failed']}  {r['median_seconds']}s")
        print(f"-> {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
