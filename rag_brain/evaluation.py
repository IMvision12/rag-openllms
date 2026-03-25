"""Evaluation framework for the RAG pipeline.

Metrics implemented:
  - Retrieval: Precision@k, Recall@k, MRR (Mean Reciprocal Rank)
  - Generation: ROUGE-L, BERTScore
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievalMetrics:
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0


@dataclass
class GenerationMetrics:
    rouge_l_f1: float = 0.0
    bert_score_f1: float = 0.0


@dataclass
class EvalResult:
    question: str
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)


def compute_retrieval_metrics(
    retrieved_sources: list[str],
    relevant_sources: list[str],
) -> RetrievalMetrics:
    """Compute retrieval quality given retrieved vs ground-truth relevant sources."""
    if not relevant_sources:
        return RetrievalMetrics()

    relevant_set = set(relevant_sources)
    hits = [src in relevant_set for src in retrieved_sources]

    true_positives = sum(hits)
    precision = true_positives / len(retrieved_sources) if retrieved_sources else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0

    # MRR: reciprocal rank of the first relevant result
    mrr = 0.0
    for i, is_hit in enumerate(hits):
        if is_hit:
            mrr = 1.0 / (i + 1)
            break

    return RetrievalMetrics(precision_at_k=precision, recall_at_k=recall, mrr=mrr)


def compute_generation_metrics(
    prediction: str,
    reference: str,
) -> GenerationMetrics:
    """Compute ROUGE-L and BERTScore between predicted answer and reference answer."""
    metrics = GenerationMetrics()

    # ROUGE-L
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        metrics.rouge_l_f1 = scores["rougeL"].fmeasure
    except ImportError:
        pass

    # BERTScore
    try:
        from bert_score import score as bert_score_fn
        _p, _r, f1 = bert_score_fn([prediction], [reference], lang="en", verbose=False)
        metrics.bert_score_f1 = f1.item()
    except ImportError:
        pass

    return metrics


def evaluate(
    question: str,
    prediction: str,
    reference: str,
    retrieved_sources: list[str],
    relevant_sources: list[str],
) -> EvalResult:
    """Full evaluation: retrieval + generation metrics."""
    return EvalResult(
        question=question,
        retrieval=compute_retrieval_metrics(retrieved_sources, relevant_sources),
        generation=compute_generation_metrics(prediction, reference),
    )
