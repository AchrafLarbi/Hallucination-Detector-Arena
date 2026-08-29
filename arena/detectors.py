"""Hallucination detectors behind one interface."""

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .text import chunk_by_tokens, make_claim, make_sentence_claims


@dataclass
class Verdict:
    detector: str
    support: float
    details: list = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""

    @property
    def hallucination_score(self) -> float:
        return 1.0 - self.support


class Detector:
    key = ""
    name = ""
    description = ""
    default_threshold = 0.5

    def score(self, context: str, question: str, answer: str) -> Verdict:
        start = time.perf_counter()
        try:
            support, details = self._score(context, question, answer)
            error = ""
        except Exception as e:
            support, details, error = math.nan, [], f"{type(e).__name__}: {e}"
        return Verdict(self.key, float(support), details, time.perf_counter() - start, error)

    def _score(self, context, question, answer):
        raise NotImplementedError


def _softmax(logits):
    import torch

    return torch.nn.functional.softmax(logits.float(), dim=-1)


class _CrossEncoderDetector(Detector):
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
        texts = [f"{c}{eos}{claim}" for c in chunks]
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


LOCAL_DETECTORS = (NLIBaseline, MiniCheck, HHEM)


def load_local_detectors() -> list:
    import torch

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return [cls() for cls in LOCAL_DETECTORS]
