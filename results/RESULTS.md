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