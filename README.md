---
title: Hallucination Detector Arena
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Can we trust LLM hallucination detectors?
startup_duration_timeout: 1h
---

<!-- The block above is configuration for Hugging Face Spaces (GitHub shows it as a table). -->

# Hallucination Detector Arena

**Can we trust LLM hallucination detectors?** Teams increasingly use automatic detectors to measure how often
their LLM "hallucinates", but a hallucination rate is only as valid as the detector that produced it. This project
audits four detectors on answers whose truth is known, across six domains, with a protocol designed to avoid
inflated numbers, and serves the results and a live playground as a web app.

**Author:** [Mohammed Achraf Larbi](https://github.com/AchrafLarbi)
&nbsp;·&nbsp; **Live app:** [huggingface.co/spaces/achraf2203/hallucination-arena](https://huggingface.co/spaces/achraf2203/hallucination-arena)

## Contents

- [Results](#results)
- [Study design](#study-design)
- [The detectors](#the-detectors)
- [The app](#the-app)
- [Reproduce the study](#reproduce-the-study)
- [Run the app locally](#run-the-app-locally)
- [Deploying on Hugging Face Spaces](#deploying-on-hugging-face-spaces)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Credits & licenses](#credits--licenses)

## Results

**Short answer: not out of the box.** On 240 held-out answers with known ground truth (results below are
generated from [`results/summary.json`](results/summary.json)):

1. **No detector is reliable overall.** The best, MiniCheck, reaches a test AUROC of **0.67** (95% CI 0.60–0.74);
   no detector exceeds a balanced accuracy of 0.61 at either threshold.
2. **Validity depends on the domain.** MiniCheck goes from AUROC **0.92** on CovidQA and 0.84 on RAGTruth to
   **0.49** on DROP and 0.51 on FinanceBench, i.e. chance level; HHEM shows the same pattern. DROP and FinanceBench
   are built around counting and arithmetic over the text, with answers that are often a single number or name;
   a plausible explanation, not tested here, is that entailment-style detectors cannot verify computed answers.
3. **Default thresholds are far from the best ones, but tuning them barely helps.** Thresholds tuned on the dev half
   are 0.08 for MiniCheck and 0.75 for HHEM instead of 0.5, yet test balanced accuracy changes by at most 0.02: the
   bottleneck is how well the detectors rank answers, not where the line is drawn.
4. **Detectors disagree.** Cohen's κ is 0.12 (NLI vs MiniCheck), 0.16 (NLI vs HHEM) and 0.50 (MiniCheck vs HHEM), and
   31 of the 240 test answers fool all three. Their accuracies at tuned thresholds are not significantly different
   (McNemar p ≥ 0.57).

**What it means in practice:** a "hallucination rate" measured with one of these detectors says as much about the
detector and the domain as about the LLM. Validate the detector on labelled examples *from your own domain*, report
ranking quality with confidence intervals, and do not compare rates measured with different detectors.

Test half: 240 answers (balanced), generated 2026-10-03 12:34.

| Detector | AUROC (95% CI) | Bal. acc. @ 0.5 | Tuned threshold | Bal. acc. @ tuned | F1 (hallucinated) | Median time |
|---|---|---|---|---|---|---|
| MiniCheck (RoBERTa-Large) | **0.672** (0.60–0.74) | 0.592 | 0.08 | 0.613 | 0.583 | 2.74 s |
| Vectara HHEM-2.1-Open | **0.638** (0.56–0.70) | 0.613 | 0.75 | 0.608 | 0.659 | 0.42 s |
| NLI baseline (DeBERTa-v3-base) | **0.576** (0.50–0.64) | 0.583 | 0.02 | 0.583 | 0.627 | 0.56 s |

AUROC per domain (test, 40 answers each):

| Detector | DROP | FinanceBench | RAGTruth | CovidQA | HaluEval | PubMedQA |
|---|---|---|---|---|---|---|
| MiniCheck (RoBERTa-Large) | 0.49 | 0.51 | 0.84 | 0.92 | 0.78 | 0.61 |
| Vectara HHEM-2.1-Open | 0.46 | 0.54 | 0.79 | 0.77 | 0.78 | 0.53 |
| NLI baseline (DeBERTa-v3-base) | 0.54 | 0.63 | 0.61 | 0.52 | 0.64 | 0.52 |

Agreement between detectors (test, tuned thresholds):

| Pair | Cohen's κ | McNemar p | Both wrong |
|---|---|---|---|
| NLI baseline vs MiniCheck | 0.12 | 0.566 | 42 / 240 |
| NLI baseline vs Vectara HHEM-2.1-Open | 0.16 | 0.606 | 50 / 240 |
| MiniCheck vs Vectara HHEM-2.1-Open | 0.50 | 1.000 | 62 / 240 |

*LLM judge (gpt-oss-120b): evaluation in progress (224/480 answers scored); it joins the comparison once it has scored every answer.*

## Study design

| Choice | What | Why |
|---|---|---|
| Data | [HaluBench](https://huggingface.co/datasets/PatronusAI/HaluBench): context + question + answer, labelled faithful / hallucinated | Ground truth across six sources: HaluEval, DROP, CovidQA, PubMedQA, FinanceBench, RAGTruth |
| Sample | 40 faithful + 40 hallucinated per source = **480 examples**, contexts ≤ 1,500 words, seed 13 | Balanced classes and domains; every detector scores the same examples |
| Split | 50/50 **dev / test** per source and label | Thresholds are tuned on dev and reported on test only |
| Ranking metric | **AUROC** with a 95% bootstrap CI (2,000 resamples) | Threshold-free: does the detector rank hallucinated answers below faithful ones? |
| Decision metric | Balanced accuracy and F1 at the **default 0.5** threshold and at the **dev-tuned** threshold | Shows how much a detector's out-of-the-box threshold costs |
| Comparison | Exact **McNemar** tests and **Cohen's κ** between detectors on the same test answers | Are differences real? Do detectors flag the same answers? |
| Claims | *question + answer*, because answers such as "0.25" or "Rams" cannot be checked alone | Same claim for every detector |

Tuning a threshold on the same data used to report accuracy is a quiet form of test-set leakage. Here the test
half is never used for any choice.

## The detectors

Every detector returns a **support score** in [0, 1] (1 = the answer is supported by the context).

| Detector | Model | How it is applied | License |
|---|---|---|---|
| NLI baseline | [`cross-encoder/nli-deberta-v3-base`](https://huggingface.co/cross-encoder/nli-deberta-v3-base) | P(entailment) of the claim, max over 400-token context chunks | Apache-2.0 |
| MiniCheck | [`lytang/MiniCheck-RoBERTa-Large`](https://huggingface.co/lytang/MiniCheck-RoBERTa-Large) (Tang et al., EMNLP 2024) | Official inference procedure (400-token chunks, `chunk</s>claim`, max over chunks), applied **per answer sentence**; answer score = minimum | MIT |
| HHEM-2.1-Open | [`vectara/hallucination_evaluation_model`](https://huggingface.co/vectara/hallucination_evaluation_model) | `predict([(context, claim)])` | Apache-2.0 |
| LLM judge | `openai/gpt-oss-120b` via [Groq](https://console.groq.com) | Asked for the probability (0–100) that the answer is fully supported; temperature 0 | API |

## The app

| Tab | What you get |
|---|---|
| **Results** | Findings computed from the data, leaderboard, AUROC per domain, default vs. tuned thresholds, agreement between detectors |
| **Try it** | Paste a context, question and answer: verdict of every detector (at the tuned thresholds), MiniCheck's sentence-by-sentence view, the LLM judge's reason |
| **Explore the study** | Browse the test answers, filter by domain or to cases where detectors disagree, open any example |
| **Method** | Data, protocol, detectors, limitations |

## Reproduce the study

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
echo "LLM_API_KEY=gsk_..." > .env         # Groq key, only for the LLM judge (never commit .env)

python run_study.py sample                # balanced 480-example sample -> results/study_sample.csv
python run_study.py score nli minicheck hhem
python run_study.py score llm_judge --out results/scores_judge.csv   # can run in parallel
python run_study.py analyze               # -> results/summary.json
python -m pytest -q                       # offline tests
```

Scoring is resumable: re-running only scores what is missing. On a laptop CPU the three local detectors take
about an hour. The LLM judge is paced for Groq's free tier, which also caps `gpt-oss-120b` at **200,000 tokens per
day** (about 220 judge calls), so the 480 answers take more than a day: re-run the same `score llm_judge` command
once the quota has refilled, then `analyze`. Until a detector has scored every answer, `analyze` lists it as
"in progress" and leaves it out of the comparison.

## Run the app locally

```bash
python app.py        # http://127.0.0.1:7860 (first start downloads ~3 GB of models)
```

## Deploying on Hugging Face Spaces

Free Gradio Spaces run on **ZeroGPU** hardware: a GPU is attached only while a `@spaces.GPU` function runs. The app
detects it (`SPACES_ZERO_GPU`), moves the three detector models to the GPU at start-up, and scores playground
requests inside a `@spaces.GPU` function. Locally it runs on CPU.

```bash
hf auth login
hf repos create hallucination-arena --type space --sdk gradio --flavor zero-a10g --secrets LLM_API_KEY
hf upload <user>/hallucination-arena . . --repo-type space \
  --exclude ".venv/*" --exclude ".env" --exclude "*__pycache__*" --exclude "data_raw/*" --exclude ".pytest_cache/*"
```

`LLM_API_KEY` (a Space *secret*) enables the LLM judge in the playground; without it the judge is simply hidden.

## Project structure

```text
hallucination-arena/
├── app.py               # Gradio app (Hugging Face Spaces entry point)
├── run_study.py         # CLI: sample, score, analyze
├── arena/
│   ├── detectors.py     # NLI baseline, MiniCheck, HHEM, LLM judge (one interface)
│   ├── data.py          # HaluBench loading, balanced dev/test sample
│   ├── metrics.py       # AUROC, thresholds, bootstrap CIs, McNemar, kappa
│   ├── study.py         # resumable scoring, analysis report
│   ├── text.py          # sentence splitting, claims, token chunking
│   ├── llm.py           # OpenAI-compatible client (pacing, retries, JSON mode)
│   └── config.py        # settings from environment variables
├── results/             # study_sample.csv, scores*.csv, summary.json (what the app shows)
├── tests/               # offline unit tests
├── requirements.txt     # app runtime (used by Spaces)
└── requirements-dev.txt # + Gradio, pyarrow, pytest
```

## Limitations

- **Sample size:** 40 test answers per domain, so per-domain AUROCs have wide uncertainty; the overall test set has 240 answers.
- **Context length:** contexts above 1,500 words are excluded (the LLM judge's API cannot take them), which removes most
  CovidQA documents; CovidQA results cover its shorter documents only.
- **Possible contamination:** HaluBench sources are public. A detector, or the LLM judge, may have seen some of them during
  training, which would inflate its scores on those sources.
- **One judge model, one prompt:** the LLM judge's results depend on both.
- **Claim construction:** *question + answer* is a reasonable, uniform choice, not the only one.

## Credits & licenses

- Code: MIT, © 2026 Mohammed Achraf Larbi (see [LICENSE](LICENSE)).
- Research framing (validate detectors before trusting their numbers) inspired by
  [LLM-hallucination-Research](https://github.com/DharambirAgrawal/LLM-hallucination-Research) by Dharambir Agrawal; no code is reused.
- Data: [HaluBench](https://huggingface.co/datasets/PatronusAI/HaluBench), Patronus AI, **CC-BY-NC-2.0** (non-commercial use).
  `results/study_sample.csv` contains a 480-example subset under the same license.
- Models: MiniCheck (Tang, Laban & Durrett, EMNLP 2024; MIT), Vectara HHEM-2.1-Open (Apache-2.0),
  `cross-encoder/nli-deberta-v3-base` (Apache-2.0).
