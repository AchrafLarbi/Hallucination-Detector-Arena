"""Hallucination Detector Arena: web app (Gradio). Hugging Face Spaces entry point; run locally with `python app.py`."""

import os

ON_ZERO_GPU = False

import json
import threading
import traceback

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import pandas as pd

from arena import config
from arena.data import SOURCE_NAMES
from arena.detectors import LLMJudge, load_local_detectors
from arena.study import SAMPLE_CSV, SUMMARY_JSON, load_scores

ORDER = ["nli", "minicheck", "hhem", "llm_judge"]
COLORS = {"nli": "#8c8c8c", "minicheck": "#2b6cb0", "hhem": "#2f855a", "llm_judge": "#805ad5"}

# ======================= Study results (precomputed by run_study.py) =======================
SUMMARY = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
DETS = {k: SUMMARY["detectors"][k] for k in ORDER if k in SUMMARY["detectors"]}
THRESHOLDS = {k: d["calibrated_threshold"] for k, d in DETS.items()}
SAMPLE = pd.read_csv(SAMPLE_CSV, dtype={"id": str}, keep_default_na=False)
WIDE = (
    load_scores().pivot(index="id", columns="detector", values="support")
    .reindex(columns=list(DETS)).reset_index()
    .merge(SAMPLE, on="id")
)
TEST = WIDE[WIDE["split"] == "test"].reset_index(drop=True)


def _correct(df: pd.DataFrame) -> pd.DataFrame:
    """Per detector: was the calibrated verdict right? (NaN when the detector failed)."""
    out = {}
    for k, thr in THRESHOLDS.items():
        pred = (df[k] < thr).astype(float).where(df[k].notna())
        out[k] = (pred == df["hallucinated"]).where(df[k].notna())
    return pd.DataFrame(out)


TEST_CORRECT = _correct(TEST)


def short(name: str) -> str:
    return name.split(" (")[0]


def in_progress_markdown() -> str:
    notes = [f"*{d['name']}: evaluation in progress ({d['n_scored']}/{d['n_total']} answers scored, limited by the "
             "LLM API's free daily token quota). It joins the comparison once it has scored every answer; meanwhile it is "
             "available live in **Try it**.*" for d in SUMMARY.get("in_progress", {}).values()]
    return "  \n".join(notes)


def findings_markdown() -> str:
    """Headline findings, computed from the results (never hand-written numbers)."""
    lines = []
    best = max(DETS, key=lambda k: DETS[k]["test_auroc"])
    b = DETS[best]
    lines.append(f"**Best at ranking answers:** {b['name']}, test AUROC **{b['test_auroc']:.2f}** "
                 f"(95% CI {b['test_auroc_ci'][0]:.2f}–{b['test_auroc_ci'][1]:.2f}).")
    gains = {k: d["test_bal_acc_calibrated"] - d["test_bal_acc_default"] for k, d in DETS.items()}
    g = max(gains, key=lambda k: abs(gains[k]))
    if abs(gains[g]) >= 0.03:
        d = DETS[g]
        change = (f"{short(d['name'])}'s balanced accuracy on the test split goes from {d['test_bal_acc_default']:.2f} "
                  f"(threshold 0.5) to {d['test_bal_acc_calibrated']:.2f} (threshold {d['calibrated_threshold']:.2f}, tuned on dev)")
        if gains[g] > 0:
            lines.append(f"**Out-of-the-box thresholds are miscalibrated:** {change}.")
        else:
            lines.append(f"**Thresholds do not transfer:** a threshold tuned on the dev split does worse on test: {change}.")
    spread = {}
    for k, d in DETS.items():
        per = {s: v["auroc"] for s, v in d["test_per_source_auroc"].items() if not np.isnan(v["auroc"])}
        if per:
            lo, hi = min(per, key=per.get), max(per, key=per.get)
            spread[k] = (per[hi] - per[lo], lo, per[lo], hi, per[hi])
    if spread:
        k = max(spread, key=lambda k: spread[k][0])
        _, lo, vlo, hi, vhi = spread[k]
        lines.append(f"**Performance depends on the domain:** {short(DETS[k]['name'])} goes from AUROC {vhi:.2f} on "
                     f"{SOURCE_NAMES.get(hi, hi)} to {vlo:.2f} on {SOURCE_NAMES.get(lo, lo)} "
                     "(40 test answers per domain, so per-domain values are uncertain).")
    best_per_source = {}
    for d in DETS.values():
        for src, v in d["test_per_source_auroc"].items():
            if not np.isnan(v["auroc"]):
                best_per_source[src] = max(best_per_source.get(src, 0.0), v["auroc"])
    hard = [s for s, a in best_per_source.items() if a < 0.65]
    if hard:
        names = ", ".join(SOURCE_NAMES.get(s, s) for s in hard)
        lines.append(f"**Hard domains:** no detector reaches AUROC 0.65 on {names}: close to chance (0.5).")
    kappas = [p["kappa"] for p in SUMMARY["pairwise"]]
    if kappas:
        span = f"κ = {kappas[0]:.2f}" if len(kappas) == 1 else f"pairwise Cohen's κ from {min(kappas):.2f} to {max(kappas):.2f}"
        verdict = "Detectors often disagree" if max(kappas) < 0.6 else "Detectors largely agree"
        lines.append(f"**{verdict}:** {span} (1 = perfect agreement); swapping detectors changes which answers get flagged.")
    complete = TEST_CORRECT.dropna()
    if len(complete):
        fooled = int((complete.sum(axis=1) == 0).sum())
        lines.append(f"**Blind spots:** {fooled} of {len(complete)} test examples fool every detector.")
    return "\n".join(f"- {line}" for line in lines)


def leaderboard() -> pd.DataFrame:
    rows = []
    for k, d in sorted(DETS.items(), key=lambda kv: -kv[1]["test_auroc"]):
        lo, hi = d["test_auroc_ci"]
        rows.append({
            "Detector": d["name"],
            "AUROC (95% CI)": f"{d['test_auroc']:.3f} ({lo:.2f}–{hi:.2f})",
            "Bal. acc. @ 0.5": f"{d['test_bal_acc_default']:.3f}",
            "Tuned threshold": f"{d['calibrated_threshold']:.2f}",
            "Bal. acc. @ tuned": f"{d['test_bal_acc_calibrated']:.3f}",
            "F1 (hallucinated)": f"{d['test_f1_calibrated']:.3f}",
            "Median time / answer": f"{d['median_seconds']:.2f} s",
        })
    return pd.DataFrame(rows)


def pairwise_table() -> pd.DataFrame:
    rows = []
    for p in SUMMARY["pairwise"]:
        rows.append({
            "Detector A": short(DETS[p["a"]]["name"]), "Detector B": short(DETS[p["b"]]["name"]),
            "Cohen's κ": f"{p['kappa']:.2f}", "McNemar p": f"{p['mcnemar_p']:.3f}",
            "Both wrong": p["both_wrong"], "n": p["n"],
        })
    return pd.DataFrame(rows)


def heatmap_figure():
    sources = list(SUMMARY["sources"])
    keys = list(DETS)
    m = np.array([[DETS[k]["test_per_source_auroc"].get(s, {}).get("auroc", np.nan) for s in sources] for k in keys])
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(keys) + 1.6))
    im = ax.imshow(m, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(sources)), [SOURCE_NAMES.get(s, s).split(" (")[0] for s in sources], rotation=20, ha="right")
    ax.set_yticks(range(len(keys)), [short(DETS[k]["name"]) for k in keys])
    for i in range(len(keys)):
        for j in range(len(sources)):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if m[i, j] > 0.82 else "#1a202c")
    fig.colorbar(im, ax=ax, label="AUROC (0.5 = chance)")
    ax.set_title("Test AUROC per domain (n = 40 per cell)", fontsize=11)
    fig.tight_layout()
    return fig


def threshold_figure():
    keys = list(DETS)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for offset, field, alpha in [(-0.18, "default", 0.45), (0.18, "calibrated", 1.0)]:
        vals = [DETS[k][f"test_bal_acc_{field}"] for k in keys]
        cis = np.array([DETS[k][f"test_bal_acc_{field}_ci"] for k in keys])
        err = np.abs(cis.T - np.array(vals))
        ax.bar(x + offset, vals, 0.34, yerr=err, capsize=3,
               color=[COLORS[k] for k in keys], alpha=alpha, edgecolor="#2d3748", linewidth=0.6)
    ax.axhline(0.5, color="#a0aec0", linestyle="--", linewidth=1)
    ax.set_xticks(x, [short(DETS[k]["name"]) for k in keys])
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Balanced accuracy (test)")
    ax.set_title("Default vs. tuned decision threshold (95% bootstrap CI)", fontsize=11)
    ax.legend(handles=[Patch(facecolor="#4a5568", alpha=0.45, label="lighter: threshold 0.5"),
                       Patch(facecolor="#4a5568", label="darker: threshold tuned on dev")],
              loc="upper left", ncol=2, fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


# ======================= Live detectors =======================
DETECTORS = []
_load_error = None
JUDGE = None
if config.LLM_API_KEY:
    from arena.llm import LLMClient

    JUDGE = LLMJudge(LLMClient(config.LLM_API_KEY, config.LLM_BASE_URL, config.LLM_JUDGE_MODEL, temperature=0.0,
                               max_queries_per_minute=config.LLM_MAX_QPM, max_tokens_per_day=config.LLM_MAX_TOKENS_PER_DAY,
                               max_retries=2, extra_body={"reasoning_effort": "low"}))


def _load_detectors():
    global _load_error
    try:
        dets = load_local_detectors()
        if ON_ZERO_GPU:
            for d in dets:
                d.model.to("cuda")  # ZeroGPU attaches the GPU inside @spaces.GPU calls
        DETECTORS.extend(dets)
    except Exception as e:
        _load_error = e
        traceback.print_exc()


def _score_local(context, question, answer):
    return [d.score(context, question, answer) for d in DETECTORS]


if ON_ZERO_GPU:
    _load_detectors()  # weights must be in place before launch on ZeroGPU
    score_local = spaces.GPU(duration=60)(_score_local)
else:
    threading.Thread(target=_load_detectors, daemon=True).start()
    score_local = _score_local


def check_answer(context, question, answer, use_judge):
    context, question, answer = (context or "").strip(), (question or "").strip(), (answer or "").strip()
    if not context or not answer:
        raise gr.Error("Fill in at least the context and the answer.")
    if len(context.split()) > config.MAX_CONTEXT_WORDS:
        raise gr.Error(f"Context too long (limit {config.MAX_CONTEXT_WORDS} words).")
    if len(answer.split()) > config.MAX_ANSWER_WORDS:
        raise gr.Error(f"Answer too long (limit {config.MAX_ANSWER_WORDS} words).")
    if _load_error is not None:
        raise gr.Error(f"Detectors failed to load: {_load_error}")
    if not DETECTORS:
        raise gr.Error("The detector models are still loading (first start downloads ~3 GB). Try again in a minute.")

    verdicts = list(score_local(context, question, answer))
    if use_judge and JUDGE is not None:
        verdicts.append(JUDGE.score(context, question, answer))

    rows, flagged, judged, sentences = [], 0, 0, []
    for v in sorted(verdicts, key=lambda v: ORDER.index(v.detector)):
        name = (DETS.get(v.detector) or SUMMARY.get("in_progress", {}).get(v.detector) or {}).get("name", v.detector)
        thr = THRESHOLDS.get(v.detector, 0.5)
        if v.error or np.isnan(v.support):
            rows.append([name, "–", f"{thr:.2f}", f"error: {v.error[:80]}", f"{v.seconds:.2f} s"])
            continue
        judged += 1
        is_hallucinated = v.support < thr
        flagged += is_hallucinated
        rows.append([name, f"{v.support:.3f}", f"{thr:.2f}",
                     "🚩 hallucination" if is_hallucinated else "✅ supported", f"{v.seconds:.2f} s"])
        if v.detector == "minicheck":
            sentences = [(s, "supported" if p >= thr else "unsupported") for s, p in v.details]
        if v.detector == "llm_judge" and v.details:
            rows[-1][3] += f": {v.details[0][0]}"

    summary = (f"### {flagged} of {judged} detectors flag a hallucination" if judged else "### No detector could score this answer")
    if judged and 0 < flagged < judged:
        summary += "\nThe detectors **disagree**: look at which one you would trust, and why."
    table = pd.DataFrame(rows, columns=["Detector", "Support score", "Threshold", "Verdict", "Time"])
    return summary, table, sentences or [(answer, None)]


# Explorer tab implemented next
# ======================= Explore the study =======================
def explore(source: str, only_disagreements: bool):
    df, correct = TEST, TEST_CORRECT
    mask = pd.Series(True, index=df.index)
    if source != "All domains":
        mask &= df["source"] == {v: k for k, v in SOURCE_NAMES.items()}.get(source, source)
    if only_disagreements:
        preds = pd.DataFrame({k: df[k] < THRESHOLDS[k] for k in DETS})
        mask &= preds.nunique(axis=1) > 1
    view = df[mask]
    table = pd.DataFrame({
        "Domain": view["source"].map(lambda s: SOURCE_NAMES.get(s, s).split(" (")[0]),
        "Truth": view["hallucinated"].map({1: "hallucinated", 0: "faithful"}),
        "Question": view["question"].str.slice(0, 90),
        "Answer": view["answer"].str.slice(0, 90),
        **{short(DETS[k]["name"]): [
            "–" if np.isnan(s) else f"{s:.2f} {'✓' if ok else '✗'}"
            for s, ok in zip(view[k], correct.loc[view.index, k])] for k in DETS},
    })
    return table, list(view["id"]), f"{len(view)} test examples (✓ = correct verdict at the tuned threshold)"


def show_example(ids, evt: gr.SelectData):
    if not ids or evt.index is None:
        return ""
    r = TEST[TEST["id"] == ids[evt.index[0]]].iloc[0]
    truth = "hallucinated" if r["hallucinated"] else "faithful"
    scores = " · ".join(f"{short(DETS[k]['name'])}: {r[k]:.2f}" for k in DETS if not np.isnan(r[k]))
    return (f"**Domain:** {SOURCE_NAMES.get(r['source'], r['source'])} · **Ground truth:** {truth}  \n"
            f"**Scores:** {scores}\n\n**Question:** {r['question']}\n\n**Answer:** {r['answer']}\n\n"
            f"**Context:**\n\n> " + r["context"].replace("\n", "\n> "))


def playground_examples():
    picks = []
    for i, src in enumerate(SUMMARY["sources"]):
        pool = TEST[(TEST["source"] == src) & (TEST["hallucinated"] == i % 2)]
        pool = pool[pool["context"].str.split().str.len() <= 350]
        if len(pool):
            r = pool.iloc[0]
            picks.append(([r["context"], r["question"], r["answer"], JUDGE is not None],
                          f"{SOURCE_NAMES.get(src, src).split(' (')[0]} · {'hallucinated' if r['hallucinated'] else 'faithful'}"))
    return [p[0] for p in picks], [p[1] for p in picks]


# ======================= UI =======================
METHOD = f"""
### Question
Hallucination detectors are increasingly used to audit LLM answers, but **are the detectors themselves valid?**
A detector you cannot trust turns every downstream "hallucination rate" into noise. This project measures four
detectors on answers whose truth is known, *before* using them for anything else.

### Data
[HaluBench](https://huggingface.co/datasets/PatronusAI/HaluBench) (Patronus AI, CC-BY-NC-2.0): context + question +
answer triples labelled faithful or hallucinated, from six sources: HaluEval, DROP, CovidQA, PubMedQA, FinanceBench
and RAGTruth. The study uses a **balanced sample of {SUMMARY['n_examples']} examples** (40 faithful + 40
hallucinated per source, contexts of at most 1,500 words), split 50/50 into **dev** and **test** halves per source and label.

### Protocol
- Every detector outputs a *support score* in [0, 1]; low support = hallucination.
- **AUROC** (threshold-free ranking quality) is reported on the test half with a 95% bootstrap CI (2,000 resamples).
- Decision thresholds are **tuned on dev only** (maximising balanced accuracy) and evaluated on test, next to the
  default 0.5, so the reported accuracy is not inflated by choosing the threshold on the evaluation data.
- Short answers ("0.25", "Rams") cannot be checked alone, so each claim is *question + answer*.
- Pairwise **McNemar** tests and **Cohen's κ** compare detectors on the same test answers.

### Detectors
| Detector | Type | License |
|---|---|---|
| NLI baseline | `cross-encoder/nli-deberta-v3-base`, P(entailment), max over context chunks | Apache-2.0 |
| MiniCheck | `lytang/MiniCheck-RoBERTa-Large` (Tang et al., EMNLP 2024), sentence-level, official chunking | MIT |
| HHEM-2.1-Open | `vectara/hallucination_evaluation_model` | Apache-2.0 |
| LLM judge | `{config.LLM_JUDGE_MODEL}` via Groq, asked for P(supported) | API |

### Limitations
- 40 test examples per domain: per-domain AUROCs have wide uncertainty.
- Very long CovidQA contexts are excluded (API limit of the LLM judge), so CovidQA results cover shorter documents only.
- HaluBench sources are public; detectors or the LLM judge may have seen some of them during training (possible
  contamination), which would inflate their scores on those sources.

### About
Developed by [Mohammed Achraf Larbi](https://github.com/AchrafLarbi). The research framing (validate detectors before
trusting them) was inspired by [LLM-hallucination-Research](https://github.com/DharambirAgrawal/LLM-hallucination-Research)
by Dharambir Agrawal; all code here is original. Results generated {SUMMARY['generated_at']}.
"""

with gr.Blocks(title="Hallucination Detector Arena", theme=gr.themes.Soft(),
               css=".note {font-size: 0.9em; color: #4a5568}") as demo:
    gr.Markdown(
        "# Hallucination Detector Arena\n"
        "**Can we trust LLM hallucination detectors?** Four detectors, measured on "
        f"{SUMMARY['n_examples']} answers with known ground truth across six domains, plus a live playground. "
        "By [Mohammed Achraf Larbi](https://github.com/AchrafLarbi)."
    )
    with gr.Tab("Results"):
        gr.Markdown(findings_markdown())
        if SUMMARY.get("in_progress"):
            gr.Markdown(in_progress_markdown(), elem_classes=["note"])
        gr.Dataframe(leaderboard(), label="Leaderboard (test half)", interactive=False, wrap=True)
        with gr.Row():
            gr.Plot(heatmap_figure(), label="Domain dependence")
        with gr.Row():
            gr.Plot(threshold_figure(), label="Thresholds")
        gr.Dataframe(pairwise_table(), label="Do detectors agree? (test half, tuned thresholds)", interactive=False)

    with gr.Tab("Try it"):
        gr.Markdown("Paste a context, a question and an answer: each detector says whether the answer is supported. "
                    "Verdicts use the thresholds tuned in the study.", elem_classes=["note"])
        with gr.Row():
            with gr.Column(scale=3):
                ctx = gr.Textbox(label="Context (source document)", lines=9, max_lines=20)
                q = gr.Textbox(label="Question")
                a = gr.Textbox(label="Answer to check", lines=3)
                use_judge = gr.Checkbox(label="Include the LLM judge (sends the text to the Groq API)",
                                        value=JUDGE is not None, interactive=JUDGE is not None)
                btn = gr.Button("Check the answer", variant="primary")
            with gr.Column(scale=2):
                verdict_md = gr.Markdown()
                verdict_table = gr.Dataframe(interactive=False, wrap=True)
                sentence_view = gr.HighlightedText(label="MiniCheck, sentence by sentence",
                                                   color_map={"supported": "green", "unsupported": "red"})
        ex_values, ex_labels = playground_examples()
        gr.Examples(ex_values, inputs=[ctx, q, a, use_judge], example_labels=ex_labels,
                    label="Examples from the test set (ground truth in the label)")
        btn.click(check_answer, [ctx, q, a, use_judge], [verdict_md, verdict_table, sentence_view], concurrency_limit=2)

    with gr.Tab("Method"):
        gr.Markdown(METHOD)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
