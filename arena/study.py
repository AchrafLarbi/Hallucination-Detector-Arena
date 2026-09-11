"""Run the detectors over the study sample (resumable) and compute the validity report."""

import csv
import itertools
import time

import numpy as np
import pandas as pd

from . import metrics as M
from .config import RESULTS_DIR
from .data import SOURCE_NAMES

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
    """Latest successful score per (id, detector) across results/scores*.csv.

    Failed attempts are kept only if the pair never succeeded (they count as failures)."""
    paths = paths or sorted(RESULTS_DIR.glob("scores*.csv"))
    s = pd.concat([pd.read_csv(p, dtype={"id": str}) for p in paths], ignore_index=True)
    s["ok"] = s["support"].notna()
    return s.sort_values("ok").drop_duplicates(["id", "detector"], keep="last").drop(columns="ok")


def _ci(metric, y, v, seed):
    lo, hi = M.bootstrap_ci(metric, y, v, seed=seed)
    return [round(lo, 4), round(hi, 4)]


def report_markdown(report: dict) -> str:
    """Results tables in Markdown, generated from summary.json (used in the README)."""
    dets = sorted(report["detectors"].values(), key=lambda d: -d["test_auroc"])
    lines = [
        f"Test half: {report['n_by_split'].get('test', 0)} answers (balanced), generated {report['generated_at']}.",
        "",
        "| Detector | AUROC (95% CI) | Bal. acc. @ 0.5 | Tuned threshold | Bal. acc. @ tuned | F1 (hallucinated) | Median time |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in dets:
        lo, hi = d["test_auroc_ci"]
        lines.append(f"| {d['name']} | **{d['test_auroc']:.3f}** ({lo:.2f}–{hi:.2f}) | {d['test_bal_acc_default']:.3f} | "
                     f"{d['calibrated_threshold']:.2f} | {d['test_bal_acc_calibrated']:.3f} | {d['test_f1_calibrated']:.3f} | "
                     f"{d['median_seconds']:.2f} s |")
    sources = list(report["sources"])
    lines += ["", "AUROC per domain (test, 40 answers each):", "",
              "| Detector | " + " | ".join(report["sources"][s].split(" (")[0] for s in sources) + " |",
              "|---|" + "---|" * len(sources)]
    for d in dets:
        cells = [f"{d['test_per_source_auroc'].get(s, {}).get('auroc', float('nan')):.2f}" for s in sources]
        lines.append(f"| {d['name']} | " + " | ".join(cells) + " |")
    names = {k: v["name"].split(" (")[0] for k, v in report["detectors"].items()}
    lines += ["", "Agreement between detectors (test, tuned thresholds):", "",
              "| Pair | Cohen's κ | McNemar p | Both wrong |", "|---|---|---|---|"]
    for p in report["pairwise"]:
        lines.append(f"| {names[p['a']]} vs {names[p['b']]} | {p['kappa']:.2f} | {p['mcnemar_p']:.3f} | {p['both_wrong']} / {p['n']} |")
    for d in report.get("in_progress", {}).values():
        lines += ["", f"*{d['name']}: evaluation in progress ({d['n_scored']}/{d['n_total']} answers scored); "
                      "it joins the comparison once it has scored every answer.*"]
    return "\n".join(lines)


def analyze(sample: pd.DataFrame, scores: pd.DataFrame, detector_names: dict) -> dict:
    data = scores.merge(sample[["id", "source", "split", "hallucinated"]].astype({"id": str}), on="id")
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "n_examples": int(len(sample)),
        "n_by_split": sample["split"].value_counts().to_dict(),
        "sources": {k: SOURCE_NAMES.get(k, k) for k in sorted(sample["source"].unique())},
        "detectors": {},
        "pairwise": [],
        "in_progress": {},  # detectors that have not scored every example yet: not compared
    }
    test_preds = {}
    for key, name in detector_names.items():
        d = data[data["detector"] == key]
        if d.empty:
            continue
        valid = d[d["support"].notna()]
        if len(valid) < len(sample):
            report["in_progress"][key] = {"name": name, "n_scored": int(len(valid)), "n_total": int(len(sample))}
            continue
        dev, test = valid[valid["split"] == "dev"], valid[valid["split"] == "test"]
        y, s = test["hallucinated"].to_numpy(), test["support"].to_numpy()
        thr = M.best_threshold(dev["hallucinated"].to_numpy(), dev["support"].to_numpy())
        pred_default, pred_cal = M.predict(s, 0.5), M.predict(s, thr)
        tp = int(((pred_cal == 1) & (y == 1)).sum())
        per_source = {}
        for src, g in test.groupby("source"):
            per_source[src] = {"auroc": round(M.auroc(g["hallucinated"], g["support"]), 4), "n": int(len(g))}
        report["detectors"][key] = {
            "name": name,
            "n_scored": int(len(valid)),
            "n_failed": int(d["support"].isna().sum()),
            "test_auroc": round(M.auroc(y, s), 4),
            "test_auroc_ci": _ci(M.auroc, y, s, seed=1),
            "default_threshold": 0.5,
            "test_bal_acc_default": round(M.balanced_accuracy(y, pred_default), 4),
            "test_bal_acc_default_ci": _ci(lambda a, b: M.balanced_accuracy(a, M.predict(b, 0.5)), y, s, seed=2),
            "calibrated_threshold": round(thr, 4),
            "test_bal_acc_calibrated": round(M.balanced_accuracy(y, pred_cal), 4),
            "test_bal_acc_calibrated_ci": _ci(lambda a, b: M.balanced_accuracy(a, M.predict(b, thr)), y, s, seed=3),
            "test_f1_calibrated": round(M.f1_hallucinated(y, pred_cal), 4),
            "test_precision_calibrated": round(tp / max(1, int(pred_cal.sum())), 4),
            "test_recall_calibrated": round(tp / max(1, int(y.sum())), 4),
            "test_per_source_auroc": per_source,
            "median_seconds": round(float(valid["seconds"].median()), 3),
        }
        test_preds[key] = test.set_index("id").assign(pred=pred_cal)[["pred", "hallucinated"]]

    for a, b in itertools.combinations(test_preds, 2):
        j = test_preds[a].join(test_preds[b], lsuffix="_a", rsuffix="_b", how="inner")
        correct_a = j["pred_a"] == j["hallucinated_a"]
        correct_b = j["pred_b"] == j["hallucinated_b"]
        report["pairwise"].append({
            "a": a, "b": b, "n": int(len(j)),
            "mcnemar_p": round(M.mcnemar_p(correct_a, correct_b), 4),
            "kappa": round(M.kappa(j["pred_a"], j["pred_b"]), 4),
            "both_wrong": int((~correct_a & ~correct_b).sum()),
        })
    return report
