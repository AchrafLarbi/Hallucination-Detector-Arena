"""Hallucination detectors behind one interface.

Every detector returns a *support* score in [0, 1]: the probability that the answer is fully
supported by the context (1 = faithful, 0 = hallucinated), plus per-unit details for display.

- NLIBaseline : DeBERTa-v3 cross-encoder, P(entailment) of question+answer, max over context chunks.
- MiniCheck   : MiniCheck-RoBERTa-Large (Tang et al., EMNLP 2024), following the official inference
                code: 400-token context chunks, "<chunk></s><claim>" input, max over chunks.
                Answers are checked sentence by sentence; the answer score is the minimum.
- HHEM        : Vectara HHEM-2.1-Open, premise = context, hypothesis = question + answer.
- LLMJudge    : an instruction-tuned LLM asked for the probability that the answer is supported.
"""

import json
import math
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .text import chunk_by_tokens, make_claim, make_sentence_claims


@dataclass
class Verdict:
    detector: str
    support: float  # P(supported); NaN when the detector failed
    details: list = field(default_factory=list)  # [(text unit, support)]
    seconds: float = 0.0
    error: str = ""

    @property
    def hallucination_score(self) -> float:
        return 1.0 - self.support


class Detector:
    key = ""
    name = ""
    description = ""
    default_threshold = 0.5  # hallucinated when support < threshold

    def score(self, context: str, question: str, answer: str) -> Verdict:
        start = time.perf_counter()
        try:
            support, details = self._score(context, question, answer)
            error = ""
        except Exception as e:  # a failing detector must not break the others
            support, details, error = math.nan, [], f"{type(e).__name__}: {e}"
        return Verdict(self.key, float(support), details, time.perf_counter() - start, error)

    def _score(self, context, question, answer):
        raise NotImplementedError


def _softmax(logits):
    import torch

    return torch.nn.functional.softmax(logits.float(), dim=-1)


class _CrossEncoderDetector(Detector):
    """Shared code for sequence-classification models scored over context chunks."""

    model_id = ""
    max_length = 512
    chunk_tokens = 400
    batch_size = 8

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).eval()
        self.torch = torch

    def _on_device(self, enc):
        return {k: v.to(self.model.device) for k, v in enc.items()}

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False, truncation=True,
                                  max_length=self.max_length)["input_ids"])

    def _chunks(self, context: str) -> list:
        return chunk_by_tokens(context, self._count_tokens, self.chunk_tokens)

    def _support_probs(self, chunks: list, claim: str) -> np.ndarray:
        raise NotImplementedError

    def _max_over_chunks(self, context: str, claim: str) -> float:
        return float(self._support_probs(self._chunks(context), claim).max())


class NLIBaseline(_CrossEncoderDetector):
    key = "nli"
    name = "NLI baseline (DeBERTa-v3-base)"
    description = "Generic natural-language-inference model: does the context entail the answer?"
    model_id = "cross-encoder/nli-deberta-v3-base"

    def __init__(self):
        super().__init__()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail_idx = labels["entailment"]

    def _support_probs(self, chunks, claim):
        out = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            enc = self.tokenizer(batch, [claim] * len(batch), truncation="only_first",
                                 max_length=self.max_length, padding=True, return_tensors="pt")
            with self.torch.no_grad():
                out.append(_softmax(self.model(**self._on_device(enc)).logits)[:, self.entail_idx].cpu().numpy())
        return np.concatenate(out)

    def _score(self, context, question, answer):
        claim = make_claim(question, answer)
        s = self._max_over_chunks(context, claim)
        return s, [(answer, s)]


class MiniCheck(_CrossEncoderDetector):
    key = "minicheck"
    name = "MiniCheck (RoBERTa-Large)"
    description = "Fact-checker trained to verify each sentence against a grounding document (EMNLP 2024)."
    model_id = "lytang/MiniCheck-RoBERTa-Large"

    def _support_probs(self, chunks, claim):
        eos = self.tokenizer.eos_token
        texts = [f"{c}{eos}{claim}" for c in chunks]  # same input format as the official code
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(texts[i:i + self.batch_size], max_length=self.max_length,
                                 truncation=True, padding=True, return_tensors="pt")
            with self.torch.no_grad():
                out.append(_softmax(self.model(**self._on_device(enc)).logits)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def _score(self, context, question, answer):
        chunks = self._chunks(context)
        claims = make_sentence_claims(question, answer)
        sentences = make_sentence_claims("", answer)
        scores = [float(self._support_probs(chunks, c).max()) for c in claims]
        return min(scores), list(zip(sentences, scores))


class HHEM(Detector):
    key = "hhem"
    name = "Vectara HHEM-2.1-Open"
    description = "Hallucination evaluation model from Vectara (T5-based, long context)."
    model_id = "vectara/hallucination_evaluation_model"

    def __init__(self):
        from transformers import AutoModelForSequenceClassification

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, trust_remote_code=True
        ).eval()

    def _score(self, context, question, answer):
        s = float(self.model.predict([(context, make_claim(question, answer))])[0])
        return s, [(answer, s)]


JUDGE_PROMPT = """You are a strict fact-checking judge for retrieval-augmented question answering.
Decide whether the ANSWER to the QUESTION is fully supported by the CONTEXT.
The answer is hallucinated if any part of it is not supported by the context or contradicts it.
Use only the context, never outside knowledge.
Return JSON only: {"support_probability": <integer 0-100, probability the answer is fully supported>, "reason": "<one short sentence>"}"""


_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
          "sixteen seventeen eighteen nineteen").split()
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i >= 2}


def _number(value) -> float:
    """Numbers, numeric strings, or English number words ("ninety", "ninety-five", "one hundred")."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().rstrip("%")
    try:
        return float(text)
    except ValueError:
        pass
    total = 0
    for word in re.split(r"[\s-]+", text):
        if word in _TENS:
            total += _TENS[word]
        elif word in _UNITS:
            total += _UNITS.index(word)
        elif word == "hundred":
            total = max(total, 1) * 100
        elif word != "and":
            raise ValueError(f"not a number: {value!r}")
    return float(total)


def parse_judge(raw: str):
    """(support in [0,1], reason) from the judge's answer; raises ValueError if unusable."""
    m = re.search(r"\{.*\}", raw or "", flags=re.S)
    data = json.loads(m.group(0) if m else raw)
    p = _number(data["support_probability"])
    if not 0 <= p <= 100:
        raise ValueError(f"support_probability out of range: {p}")
    return p / 100.0, str(data.get("reason", ""))


class LLMJudge(Detector):
    key = "llm_judge"
    description = "A large language model asked to grade the answer against the context."

    def __init__(self, client):
        self.client = client
        self.name = f"LLM judge ({client.model_name.split('/')[-1]})"

    def _score(self, context, question, answer):
        payload = json.dumps({"CONTEXT": context, "QUESTION": question, "ANSWER": answer}, ensure_ascii=False)
        support, reason = parse_judge(self.client.query(payload, JUDGE_PROMPT))
        return support, [(reason or answer, support)]


LOCAL_DETECTORS = (NLIBaseline, MiniCheck, HHEM)


def load_local_detectors() -> list:
    import torch

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return [cls() for cls in LOCAL_DETECTORS]
