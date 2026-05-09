"""
RAGTruth Grader for evaluating hallucination detection quality.

This module evaluates the output of ``UnifiedGroundednessScorer`` (or any scorer
that produces ``ClaimVerdict`` objects with character offsets) against the
RAGTruth dataset ground truth annotations.

Evaluation approach
-------------------

**Claim-level evaluation** (primary):

    Ground truth hallucination spans from RAGTruth have ``{start, end, text, label_type}``.
    Predicted claims have ``{anchor_text, start_offset, end_offset, verdict}``.

    A predicted claim is considered a **hallucination detection** if its verdict
    is ``"baseless"`` or ``"contradicted"`` (i.e. anything other than ``"supported"``).

    Matching is done via **character-level span overlap**: a predicted hallucination
    claim is matched to a ground truth span if their character ranges overlap.

    - **Recall**: fraction of GT hallucination spans that are overlapped by at least
      one predicted hallucination claim.
    - **Precision**: fraction of predicted hallucination claims that overlap at least
      one GT hallucination span.

    RAGTruth distinguishes "Evident" and "Subtle" types — we merge them into one
    category (any hallucination = positive).

**Response-level evaluation** (secondary):

    A response is labeled as "contains hallucination" if it has any GT hallucination
    span. The scorer produces a response-level score (0–1). We compute AUC-ROC
    using ``1 - response_score`` as the predicted probability of hallucination.

**Confusion matrix** (optional):

    For matched GT↔predicted pairs, we can compare the type mapping:
    - GT "Evident Conflict" / "Subtle Conflict" → expected predicted verdict "contradicted"
    - GT "Evident Baseless Info" / "Subtle Baseless Info" → expected predicted verdict "baseless"
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAGTruth label type → binary hallucination category
# ---------------------------------------------------------------------------

#: Map RAGTruth label_type to a simplified category.
#: We merge "Evident" and "Subtle" variants into one.
RAGTRUTH_LABEL_TO_CATEGORY: Dict[str, str] = {
    "Evident Conflict": "conflict",
    "Subtle Conflict": "conflict",
    "Evident Baseless Info": "baseless",
    "Subtle Baseless Info": "baseless",
}


def _spans_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    """Check if half-open intervals [s1, e1) and [s2, e2) overlap."""
    return s1 < e2 and s2 < e1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClaimLevelMetrics:
    """
    Claim-level evaluation metrics.

    Attributes
    ----------
    recall : float
        Fraction of GT hallucination spans matched by at least one predicted
        hallucination claim.
    precision : float
        Fraction of predicted hallucination claims that match at least one
        GT hallucination span.
    f1 : float
        Harmonic mean of precision and recall.
    n_gt_spans : int
        Total number of ground truth hallucination spans.
    n_pred_hallucinations : int
        Total number of predicted hallucination claims (baseless + contradicted).
    n_gt_matched : int
        Number of GT spans matched by at least one prediction.
    n_pred_matched : int
        Number of predicted hallucination claims matched by at least one GT span.
    """

    recall: float
    precision: float
    f1: float
    n_gt_spans: int
    n_pred_hallucinations: int
    n_gt_matched: int
    n_pred_matched: int


@dataclass
class ResponseLevelMetrics:
    """
    Response-level evaluation metrics.

    Attributes
    ----------
    auc_roc : float
        AUC-ROC using ``1 - response_score`` as predicted hallucination probability.
        NaN if sklearn is not available or if there is only one class.
    n_responses : int
        Total number of responses evaluated.
    n_hallucinated : int
        Number of responses with at least one GT hallucination span.
    n_clean : int
        Number of responses with no GT hallucination spans.
    """

    auc_roc: float
    n_responses: int
    n_hallucinated: int
    n_clean: int


@dataclass
class ConfusionEntry:
    """
    A single matched pair for the confusion matrix analysis.

    Attributes
    ----------
    gt_label_type : str
        Original RAGTruth label_type (e.g. "Evident Conflict").
    gt_category : str
        Simplified category: "conflict" or "baseless".
    predicted_verdict : str
        The predicted verdict: "baseless" or "contradicted".
    gt_text : str
        The ground truth hallucination text span.
    predicted_anchor : str
        The predicted anchor_text.
    answer_idx : int
        Index of the answer in the input list.
    """

    gt_label_type: str
    gt_category: str
    predicted_verdict: str
    gt_text: str
    predicted_anchor: str
    answer_idx: int


@dataclass
class RAGTruthEvalResult:
    """
    Full evaluation result from ``RAGTruthGrader.evaluate()``.

    Attributes
    ----------
    claim_level : ClaimLevelMetrics
        Claim-level precision, recall, F1.
    response_level : ResponseLevelMetrics
        Response-level AUC-ROC.
    confusion_entries : List[ConfusionEntry]
        Per-match details for confusion matrix analysis.
    per_answer_details : List[Dict[str, Any]]
        Per-answer breakdown with GT spans, predicted hallucinations, matches.
    """

    claim_level: ClaimLevelMetrics
    response_level: ResponseLevelMetrics
    confusion_entries: List[ConfusionEntry] = field(default_factory=list)
    per_answer_details: List[Dict[str, Any]] = field(default_factory=list)

    def confusion_matrix_summary(self) -> Dict[str, Dict[str, int]]:
        """
        Build a confusion matrix: GT category → predicted verdict → count.

        Returns
        -------
        Dict[str, Dict[str, int]]
            E.g. ``{"conflict": {"contradicted": 5, "baseless": 2}, ...}``
        """
        matrix: Dict[str, Dict[str, int]] = {}
        for entry in self.confusion_entries:
            cat = entry.gt_category
            pred = entry.predicted_verdict
            if cat not in matrix:
                matrix[cat] = {}
            matrix[cat][pred] = matrix[cat].get(pred, 0) + 1
        return matrix


# ---------------------------------------------------------------------------
# Main grader class
# ---------------------------------------------------------------------------


class RAGTruthGrader:
    """
    Evaluates hallucination detection quality against RAGTruth ground truth.

    This grader takes the output of ``UnifiedGroundednessScorer`` (specifically
    the ``claim_verdicts`` with character offsets) and compares it against
    RAGTruth's annotated hallucination spans.

    Parameters
    ----------
    overlap_mode : str, default="any"
        How to determine if a predicted claim matches a GT span.
        Currently only ``"any"`` (any character overlap) is supported.
    """

    def __init__(self, overlap_mode: str = "any") -> None:
        self.overlap_mode = overlap_mode

    def evaluate(
        self,
        gt_hallucination_labels: List[List[Dict[str, Any]]],
        claim_verdicts: List[List[Any]],
        response_scores: Optional[List[float]] = None,
    ) -> RAGTruthEvalResult:
        """
        Evaluate predicted claim verdicts against RAGTruth ground truth.

        Parameters
        ----------
        gt_hallucination_labels : List[List[Dict[str, Any]]]
            Per-answer list of ground truth hallucination spans.
            Each span is a dict with keys: ``start``, ``end``, ``text``, ``label_type``.
            For clean answers, pass an empty list ``[]``.

        claim_verdicts : List[List[Any]]
            Per-answer list of predicted claim records from the scorer.
            Each item may be either an object with ``verdict``, ``start_offset``,
            ``end_offset``, ``anchor_text`` and ``claim`` attributes, or a dict with
            the same keys.

        response_scores : List[float], optional
            Per-answer response-level scores (0–1) from the scorer.
            If provided, response-level AUC-ROC is computed.

        Returns
        -------
        RAGTruthEvalResult
            Full evaluation result with claim-level and response-level metrics.
        """
        if len(gt_hallucination_labels) != len(claim_verdicts):
            raise ValueError(
                f"Length mismatch: gt_hallucination_labels={len(gt_hallucination_labels)}, "
                f"claim_verdicts={len(claim_verdicts)}"
            )

        # Accumulators
        total_gt_spans = 0
        total_gt_matched = 0
        total_pred_hallucinations = 0
        total_pred_matched = 0
        confusion_entries: List[ConfusionEntry] = []
        per_answer_details: List[Dict[str, Any]] = []

        for i, (gt_spans, verdicts) in enumerate(
            zip(gt_hallucination_labels, claim_verdicts)
        ):
            detail = self._evaluate_single(
                gt_spans=gt_spans,
                verdicts=verdicts,
                answer_idx=i,
            )
            per_answer_details.append(detail)

            total_gt_spans += detail["n_gt_spans"]
            total_gt_matched += detail["n_gt_matched"]
            total_pred_hallucinations += detail["n_pred_hallucinations"]
            total_pred_matched += detail["n_pred_matched"]
            confusion_entries.extend(detail["confusion_entries"])

        # Claim-level metrics
        recall = total_gt_matched / total_gt_spans if total_gt_spans > 0 else 1.0
        precision = (
            total_pred_matched / total_pred_hallucinations
            if total_pred_hallucinations > 0
            else 1.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        claim_metrics = ClaimLevelMetrics(
            recall=recall,
            precision=precision,
            f1=f1,
            n_gt_spans=total_gt_spans,
            n_pred_hallucinations=total_pred_hallucinations,
            n_gt_matched=total_gt_matched,
            n_pred_matched=total_pred_matched,
        )

        # Response-level metrics
        response_metrics = self._compute_response_level(
            gt_hallucination_labels=gt_hallucination_labels,
            response_scores=response_scores,
        )

        return RAGTruthEvalResult(
            claim_level=claim_metrics,
            response_level=response_metrics,
            confusion_entries=confusion_entries,
            per_answer_details=per_answer_details,
        )

    def _evaluate_single(
        self,
        gt_spans: List[Dict[str, Any]],
        verdicts: List[Any],
        answer_idx: int,
    ) -> Dict[str, Any]:
        """
        Evaluate a single answer's predicted verdicts against its GT spans.

        Returns a dict with per-answer metrics and match details.
        """
        # Filter predicted hallucination claims (non-supported, with valid offsets)
        pred_hallucinations = [
            cv
            for cv in verdicts
            if self._get_claim_field(cv, "verdict") in ("baseless", "contradicted")
            and self._get_claim_field(cv, "start_offset", -1) >= 0
        ]

        # Filter valid GT spans
        valid_gt_spans = [
            s
            for s in gt_spans
            if isinstance(s, dict)
            and "start" in s
            and "end" in s
            and s.get("label_type", "") in RAGTRUTH_LABEL_TO_CATEGORY
        ]

        n_gt_spans = len(valid_gt_spans)
        n_pred_hallucinations = len(pred_hallucinations)

        # Track which GT spans and predictions are matched
        gt_matched = [False] * n_gt_spans
        pred_matched = [False] * n_pred_hallucinations
        confusion_entries: List[ConfusionEntry] = []

        for gi, gt_span in enumerate(valid_gt_spans):
            gt_start = gt_span["start"]
            gt_end = gt_span["end"]
            gt_label_type = gt_span.get("label_type", "")
            gt_category = RAGTRUTH_LABEL_TO_CATEGORY.get(gt_label_type, "unknown")
            gt_text = gt_span.get("text", "")

            for pi, pred in enumerate(pred_hallucinations):
                pred_start = self._get_claim_field(pred, "start_offset", -1)
                pred_end = self._get_claim_field(pred, "end_offset", -1)
                pred_verdict = self._get_claim_field(pred, "verdict", "unknown")
                pred_anchor = self._get_claim_field(pred, "anchor_text", "")

                if _spans_overlap(pred_start, pred_end, gt_start, gt_end):
                    gt_matched[gi] = True
                    pred_matched[pi] = True

                    confusion_entries.append(
                        ConfusionEntry(
                            gt_label_type=gt_label_type,
                            gt_category=gt_category,
                            predicted_verdict=pred_verdict,
                            gt_text=gt_text,
                            predicted_anchor=pred_anchor,
                            answer_idx=answer_idx,
                        )
                    )

        n_gt_matched = sum(gt_matched)
        n_pred_matched = sum(pred_matched)

        return {
            "answer_idx": answer_idx,
            "n_gt_spans": n_gt_spans,
            "n_pred_hallucinations": n_pred_hallucinations,
            "n_gt_matched": n_gt_matched,
            "n_pred_matched": n_pred_matched,
            "gt_spans": valid_gt_spans,
            "pred_hallucinations": [
                {
                    "verdict": self._get_claim_field(cv, "verdict", "unknown"),
                    "anchor_text": self._get_claim_field(cv, "anchor_text", ""),
                    "start": self._get_claim_field(cv, "start_offset", -1),
                    "end": self._get_claim_field(cv, "end_offset", -1),
                    "claim": self._get_claim_field(cv, "claim", ""),
                    "matched": pred_matched[pi],
                }
                for pi, cv in enumerate(pred_hallucinations)
            ],
            "confusion_entries": confusion_entries,
        }

    @staticmethod
    def _get_claim_field(claim: Any, field: str, default: Any = None) -> Any:
        """Read a claim field from either an object or a dict."""
        if isinstance(claim, dict):
            return claim.get(field, default)
        return getattr(claim, field, default)

    @staticmethod
    def _compute_response_level(
        gt_hallucination_labels: List[List[Dict[str, Any]]],
        response_scores: Optional[List[float]],
    ) -> ResponseLevelMetrics:
        """
        Compute response-level AUC-ROC.

        A response is positive (hallucinated) if it has any GT hallucination span.
        The predicted score is ``1 - response_score`` (higher = more likely hallucinated).
        """
        gt_binary = [
            1
            if any(
                isinstance(s, dict)
                and s.get("label_type", "") in RAGTRUTH_LABEL_TO_CATEGORY
                for s in spans
            )
            else 0
            for spans in gt_hallucination_labels
        ]

        n_responses = len(gt_binary)
        n_hallucinated = sum(gt_binary)
        n_clean = n_responses - n_hallucinated

        auc_roc = np.nan

        if response_scores is not None and n_hallucinated > 0 and n_clean > 0:
            try:
                from sklearn.metrics import roc_auc_score

                # Higher hallucination probability = lower response_score
                pred_proba = [1.0 - s for s in response_scores]
                auc_roc = float(roc_auc_score(gt_binary, pred_proba))
            except ImportError:
                logger.warning(
                    "sklearn not available, skipping AUC-ROC computation"
                )
            except ValueError as exc:
                logger.warning("AUC-ROC computation failed: %s", exc)

        return ResponseLevelMetrics(
            auc_roc=auc_roc,
            n_responses=n_responses,
            n_hallucinated=n_hallucinated,
            n_clean=n_clean,
        )
