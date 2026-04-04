"""
Context Groundedness Scorer for RAG hallucination detection.

This module provides a scorer that verifies whether claims in a generated answer
are grounded in the provided context.

Pipeline:
    1. Decompose answer into atomic claims (via ResponseDecomposer)
    2. Verify each claim against context (via EntailmentClassifier)
    3. Aggregate claim-level scores into response-level scores
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from rich.progress import Progress, TextColumn
from rich.errors import LiveError

from uqlm.longform.decomposition import ResponseDecomposer
from uqlm.nli.entailment import EntailmentClassifier
from uqlm.utils.display import (
    ConditionalBarColumn,
    ConditionalTimeElapsedColumn,
    ConditionalTextColumn,
    ConditionalSpinnerColumn,
)

logger = logging.getLogger(__name__)


@dataclass
class ContextGroundednessResult:
    """
    Result of context groundedness scoring.

    Attributes
    ----------
    queries : List[str]
        The original queries.
    contexts : List[str]
        The contexts (retrieved documents) used for grounding verification.
    answers : List[str]
        The generated answers that were scored.
    claim_sets : List[List[str]]
        Atomic claims extracted from each answer.
    claim_labels : List[List[str]]
        Semantic label per claim: "supported", "contradiction", "baseless", or "unknown".
    claim_scores : List[List[float]]
        Numeric score per claim: 1.0 (supported), 0.5 (baseless/neutral), 0.0 (contradiction).
    response_scores : List[float]
        Aggregated score per response.
    metadata : dict
        Scoring metadata (style, aggregation, etc.).
    """

    queries: List[str]
    contexts: List[str]
    answers: List[str]
    claim_sets: List[List[str]]
    claim_labels: List[List[str]]
    claim_scores: List[List[float]]
    response_scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextGroundednessScorer:
    """
    Scores RAG answers by verifying claim groundedness against the provided context.

    This scorer decomposes the answer into atomic claims, then checks each claim
    against the context using LLM-based NLI. No sampled responses are needed —
    the context itself serves as the ground truth source.

    Parameters
    ----------
    nli_llm : BaseChatModel
        LLM for NLI verification. Required — will raise ValueError if not provided.

    claim_decomposition_llm : BaseChatModel, optional
        LLM for decomposing answers into claims. If None, ``nli_llm`` is used.

    entailment_style : str, default="nli_classification"
        Prompt style for entailment evaluation. Supported values:
        - "nli_classification": 3-class (entailment/contradiction/neutral)
        - "binary": 2-class (yes/no)
        - "p_true", "p_false", "p_neutral": specialized binary prompts

    aggregation : str, default="mean"
        How to aggregate claim-level scores to response-level. Must be "mean" or "min".
    """

    def __init__(
        self,
        nli_llm: BaseChatModel,
        claim_decomposition_llm: Optional[BaseChatModel] = None,
        entailment_style: str = "nli_classification",
        aggregation: str = "mean",
    ) -> None:
        if nli_llm is None:
            raise ValueError(
                "nli_llm is required for ContextGroundednessScorer. "
                "Provide a LangChain BaseChatModel instance."
            )

        if aggregation not in ("mean", "min"):
            raise ValueError(
                f"Invalid aggregation: {aggregation!r}. Must be 'mean' or 'min'."
            )

        self.nli_llm = nli_llm
        self.entailment_style = entailment_style
        self.aggregation = aggregation

        decomposition_llm = claim_decomposition_llm if claim_decomposition_llm else nli_llm
        self.decomposer = ResponseDecomposer(claim_decomposition_llm=decomposition_llm)
        self.entailment_classifier = EntailmentClassifier(
            nli_llm=nli_llm, style=entailment_style
        )

        logger.debug(
            "ContextGroundednessScorer initialized with style=%s, aggregation=%s",
            entailment_style,
            aggregation,
        )

    async def score(
        self,
        queries: List[str],
        contexts: List[str],
        answers: List[str],
        return_raw_judge_responses: bool = False,
        show_progress_bars: bool = True,
    ) -> ContextGroundednessResult:
        """
        Score pre-generated answers by verifying claim groundedness against contexts.

        Parameters
        ----------
        queries : List[str]
            The original queries.
        contexts : List[str]
            The contexts (retrieved documents) for grounding verification.
        answers : List[str]
            The generated answers to score.
        return_raw_judge_responses : bool, default=False
            If True, includes raw LLM judge responses in result metadata.
        show_progress_bars : bool, default=True
            If True, displays rich progress bars during decomposition and verification.

        Returns
        -------
        ContextGroundednessResult
            Contains claims, labels, scores, and aggregated response scores.

        Raises
        ------
        ValueError
            If input lists have mismatched lengths.
        """
        if not (len(queries) == len(contexts) == len(answers)):
            raise ValueError(
                f"Input lists must have equal length. "
                f"Got queries={len(queries)}, contexts={len(contexts)}, answers={len(answers)}."
            )

        logger.debug("Scoring %d answers", len(answers))

        progress_bar = self._build_progress_bar(show_progress_bars)

        try:
            if progress_bar:
                progress_bar.add_task("✂️  Decomposition")

            # Step 1: Decompose answers into claims
            claim_sets = await self._decompose_answers(answers, progress_bar=progress_bar)
            logger.debug(
                "Decomposed %d answers into claims (total claims: %d)",
                len(answers),
                sum(len(cs) for cs in claim_sets),
            )

            if progress_bar:
                progress_bar.add_task("")
                progress_bar.add_task("📈 Verification")

            # Step 2: Verify each claim against its corresponding context
            if return_raw_judge_responses:
                claim_scores, claim_labels, raw_responses = await self._verify_claims_against_contexts(
                    claim_sets=claim_sets, contexts=contexts,
                    return_raw_judge_responses=True, progress_bar=progress_bar,
                )
            else:
                claim_scores, claim_labels = await self._verify_claims_against_contexts(
                    claim_sets=claim_sets, contexts=contexts, progress_bar=progress_bar,
                )
            logger.debug("Verification complete")

        finally:
            if progress_bar:
                progress_bar.stop()

        # Step 3: Aggregate claim scores to response scores
        response_scores = self._aggregate_scores(claim_scores)

        return ContextGroundednessResult(
            queries=queries,
            contexts=contexts,
            answers=answers,
            claim_sets=claim_sets,
            claim_labels=claim_labels,
            claim_scores=claim_scores,
            response_scores=response_scores,
            metadata={
                "entailment_style": self.entailment_style,
                "aggregation": self.aggregation,
                "raw_judge_responses": raw_responses if return_raw_judge_responses else None,
            },
        )

    @staticmethod
    def _build_progress_bar(show_progress_bars: bool) -> Optional[Progress]:
        """Create and start a rich Progress bar, or return None if disabled."""
        if not show_progress_bars:
            return None
        try:
            completion_text = "[progress.percentage]{task.completed}/{task.total}"
            progress_bar = Progress(
                ConditionalSpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                ConditionalBarColumn(),
                ConditionalTextColumn(completion_text),
                ConditionalTimeElapsedColumn(),
            )
            progress_bar.start()
            return progress_bar
        except LiveError:
            logger.debug("Could not create progress bar (LiveError), continuing without it")
            return None

    async def _decompose_answers(
        self, answers: List[str], progress_bar: Optional[Progress] = None
    ) -> List[List[str]]:
        """Decompose answers into atomic claims using ResponseDecomposer."""
        claim_sets = await self.decomposer.decompose_claims(responses=answers, progress_bar=progress_bar)
        return claim_sets

    async def _verify_claims_against_contexts(
        self,
        claim_sets: List[List[str]],
        contexts: List[str],
        return_raw_judge_responses: bool = False,
        progress_bar: Optional[Progress] = None,
    ):
        """
        Verify each claim against its corresponding context via EntailmentClassifier.

        For each answer i, every claim in claim_sets[i] is checked against contexts[i].

        Returns
        -------
        claim_scores : List[List[float]]
            Numeric scores per claim.
        claim_labels : List[List[str]]
            Semantic labels per claim.
        raw_responses : List[List[str]]
            Raw LLM judge responses per claim (only if return_raw_judge_responses=True).
        """
        # Flatten all (context, claim) pairs for batched processing
        flat_premises: List[str] = []
        flat_hypotheses: List[str] = []
        # Track structure for unflattening: (answer_idx, claim_idx)
        structure: List[tuple[int, int]] = []

        for i, (claim_set, context) in enumerate(zip(claim_sets, contexts)):
            for j, claim in enumerate(claim_set):
                flat_premises.append(context)
                flat_hypotheses.append(claim)
                structure.append((i, j))

        if not flat_premises:
            # No claims to verify — return empty results
            empty = [[] for _ in claim_sets]
            if return_raw_judge_responses:
                return [[] for _ in claim_sets], [[] for _ in claim_sets], [[] for _ in claim_sets]
            return [[] for _ in claim_sets], [[] for _ in claim_sets]

        logger.debug("Verifying %d claim-context pairs", len(flat_premises))

        # Call EntailmentClassifier in batch
        nli_result = await self.entailment_classifier.judge_entailment(
            premises=flat_premises,
            hypotheses=flat_hypotheses,
            return_labels=True,
            progress_bar=progress_bar,
        )

        flat_scores = nli_result["scores"]
        flat_labels = nli_result["labels"]

        # Unflatten back to per-answer structure
        claim_scores: List[List[float]] = [[] for _ in claim_sets]
        claim_labels: List[List[str]] = [[] for _ in claim_sets]

        for idx, (answer_i, claim_j) in enumerate(structure):
            claim_scores[answer_i].append(flat_scores[idx])
            claim_labels[answer_i].append(flat_labels[idx])

        if return_raw_judge_responses:
            flat_raw_responses = nli_result["judge_responses"]
            raw_responses: List[List[str]] = [[] for _ in claim_sets]
            for idx, (answer_i, claim_j) in enumerate(structure):
                raw_responses[answer_i].append(flat_raw_responses[idx])
            return claim_scores, claim_labels, raw_responses

        return claim_scores, claim_labels

    def _aggregate_scores(self, claim_scores: List[List[float]]) -> List[float]:
        """
        Aggregate claim-level scores to response-level scores.

        Handles NaN values by ignoring them. If all claims are NaN,
        the response score is NaN.
        """
        response_scores = []
        for scores in claim_scores:
            if not scores:
                response_scores.append(np.nan)
                continue

            valid_scores = [s for s in scores if not np.isnan(s)]
            if not valid_scores:
                response_scores.append(np.nan)
                continue

            if self.aggregation == "mean":
                response_scores.append(float(np.mean(valid_scores)))
            elif self.aggregation == "min":
                response_scores.append(float(np.min(valid_scores)))

        return response_scores
