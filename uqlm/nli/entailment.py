import math
from typing import Any, Dict, Optional, List, Tuple

import asyncio
import numpy as np
import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from rich.progress import Progress
from uqlm.utils.prompts.entailment_prompts import get_entailment_prompt


SYSTEM_PROMPT = "You are a helpful assistant that evaluates natural language inference relationships."
STR_SCORE_MAP = {"yes": 1.0, "no": 0.0}

# Mapping for nli_classification style: entailment/contradiction/neutral → numeric scores
NLI_LABEL_SCORE_MAP = {"entailment": 1.0, "contradiction": 0.0, "neutral": 0.5}
# Mapping for nli_classification style: entailment/contradiction/neutral → semantic labels
NLI_LABEL_MAP = {"entailment": "supported", "contradiction": "contradiction", "neutral": "baseless"}


class EntailmentClassifier:
    def __init__(self, nli_llm: Optional[BaseChatModel] = None, style: str = "binary") -> None:
        """
        A class to compute NLI predictions.

        Parameters
        ----------
        nli_llm : BaseChatModel, default=None
            A LangChain chat model for LLM-based NLI inference. If provided, takes precedence over nli_model_name.

        style : str, default="binary"
            The entailment prompt style. Supported values: "binary", "nli_classification",
            "p_true", "p_false", "p_neutral". Controls both prompt construction and score extraction.
        """
        self.nli_llm = nli_llm
        self.style = style
        self.completed = 0
        self.num_responses = None

    async def judge_entailment(self, premises: List[str], hypotheses: List[str], retries: int = 5, progress_bar: Optional[Progress] = None, return_labels: bool = False) -> Dict[str, Any]:
        """
        Async version of predict() for single NLI prediction.

        This method computes NLI predictions on the provided inputs asynchronously.
        For LangChain models, this enables concurrent LLM calls which significantly improves performance.
        For HuggingFace models, this wraps the synchronous call for API consistency.

        Parameters
        ----------
        premises : List[str]
            The premise texts for NLI classification.

        claims : List[str]
            The hypothesis texts for NLI classification.

        retries : int, default=5
            Number of times to retry for failed score extraction

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses

        Returns
        -------
        Dict[str, Any]
            The entailment prompts, raw LLM outputs, and extracted entailment/contradiction scores
        """
        prompts = self._construct_prompts(premises=premises, hypotheses=hypotheses)
        self.num_prompts = len(prompts)
        if progress_bar:
            total = self.num_prompts if not self.num_responses else self.num_responses
            self.progress_task = progress_bar.add_task("  - Evaluating claim entailment...", total=total)

        tasks = [self._evaluate_claim_response_pair(prompt, progress_bar=progress_bar) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        scores = self._extract_scores(responses)
        if return_labels:
            labels = self._extract_labels(responses)
            df = pd.DataFrame({"judge_prompts": prompts, "judge_responses": responses, "scores": scores, "labels": labels})
        else:
            df = pd.DataFrame({"judge_prompts": prompts, "judge_responses": responses, "scores": scores})

        # Retry logic for failed extractions
        retry = 0
        while retry <= retries:
            retry += 1

            # Find any failures
            score_failures = df[pd.isna(df.scores)]

            # If ANY failures exist, retry
            if len(score_failures) > 0:
                # Get all failure indices
                failure_indices = set(score_failures.index)

                tasks_tmp = [self._evaluate_claim_response_pair(prompt) for prompt in list(df.loc[list(failure_indices)])]
                response_tmp = await asyncio.gather(*tasks_tmp)

                retry_data = self._extract_scores(response_tmp["data"]["response"])

                df.loc[list(failure_indices), "scores"] = retry_data

            # Exit if no more failures
            if len(score_failures) == 0:
                break
        return {col: list(df[col]) for col in df.columns}

    async def evaluate_claim_entailment(self, response_sets: List[List[str]], claim_sets: List[List[str]], retries: int = 5, progress_bar: Optional[Progress] = None) -> List[np.ndarray]:
        """
        Implements self.judge_entailment for claim-response pairs and reformats result in List[np.array]

        Parameters
        ----------
        response_sets : List[List[str]]
            The premise texts for NLI classification.

        claim_sets : List[List[str]]
            The hypothesis texts for NLI classification.

        retries : int, default=5
            Number of times to retry for failed score extraction

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses

        Returns
        -------
        List[np.ndarray]
            Entailment and contradiction scores
        """
        self.num_responses = len(response_sets)
        flat_responses, flat_claims, indices, shapes = self._flatten_inputs(response_sets=response_sets, claim_sets=claim_sets)
        nli_result = await self.judge_entailment(hypotheses=flat_claims, premises=flat_responses, retries=retries, progress_bar=progress_bar)
        entailment_score_lists = self._format_result_arrays(flat_predictions=nli_result["scores"], indices=indices, shapes=shapes)
        return entailment_score_lists

    def _extract_scores(self, judge_responses: List[str]) -> List[float]:
        """Map entailment judge responses to numerical scores"""
        return [self._extract_single_score(text, style=self.style) for text in judge_responses]

    def _extract_labels(self, judge_responses: List[str]) -> List[str]:
        """Map entailment judge responses to semantic labels"""
        return [self._extract_single_label(text, style=self.style) for text in judge_responses]

    async def _evaluate_claim_response_pair(self, prompt: str, progress_bar: Optional[Progress] = None) -> str:
        """Decompose single response into claims using LLM and extract claims from the result"""
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]
        generation = await self.nli_llm.ainvoke(messages)
        response = generation.content

        if progress_bar:
            if self.num_responses:
                self.completed += 1
                progress_bar.update(self.progress_task, completed=math.floor(self.num_responses * self.completed / self.num_prompts))
            else:
                progress_bar.update(self.progress_task, advance=1)
        return response

    @staticmethod
    def _extract_single_score(response_text: str, style: str = "binary") -> float:
        """
        Map response text to numeric score.

        For binary styles (binary, p_true, p_false, p_neutral): yes→1.0, no→0.0
        For nli_classification: entailment→1.0, neutral→0.5, contradiction→0.0
        """
        clean_text = response_text.strip().lower()

        if style == "nli_classification":
            score_map = NLI_LABEL_SCORE_MAP
        else:
            score_map = STR_SCORE_MAP

        for word, score in score_map.items():
            # Best: response starts with the value
            if clean_text.startswith(word):
                return score

        for word, score in score_map.items():
            # fallback: substring search
            if word in clean_text:
                return score

        return np.nan

    @staticmethod
    def _extract_single_label(response_text: str, style: str = "binary") -> str:
        """
        Map response text to a semantic label string.

        For binary styles: "supported" or "not_supported"
        For nli_classification: "supported", "contradiction", or "baseless"
        """
        clean_text = response_text.strip().lower()

        if style == "nli_classification":
            for word, label in NLI_LABEL_MAP.items():
                if clean_text.startswith(word):
                    return label
            for word, label in NLI_LABEL_MAP.items():
                if word in clean_text:
                    return label
            return "unknown"
        else:
            for word in ("yes",):
                if clean_text.startswith(word) or word in clean_text:
                    return "supported"
            for word in ("no",):
                if clean_text.startswith(word) or word in clean_text:
                    return "not_supported"
            return "unknown"

    def _construct_prompts(self, premises: List[str], hypotheses: List[str]) -> List[str]:
        """Construct prompt for entailment evaluation using the configured style"""
        return [get_entailment_prompt(claim=hypotheses[i], source_text=premises[i], style=self.style) for i in range(len(premises))]

    @staticmethod
    def _flatten_inputs(response_sets: List[List[str]], claim_sets: List[List[str]]) -> Tuple[List[str], List[str], List[int], List[np.ndarray]]:
        """
        Flattens nested premises and hypotheses for processing and provides mapping
        to reconstruct the original structure.
        """
        flat_responses = []
        flat_claims = []
        indices = []
        shapes = []

        for i in range(len(response_sets)):
            response_list = response_sets[i]
            claim_list = claim_sets[i]
            shapes.append((len(claim_list), len(response_list)))

            for j, claim in enumerate(claim_list):
                for k, response in enumerate(response_list):
                    flat_responses.append(response)
                    flat_claims.append(claim)
                    indices.append((i, j, k))

        return flat_responses, flat_claims, indices, shapes

    @staticmethod
    def _format_result_arrays(flat_predictions: List[int], indices: List[Tuple[int, int, int]], shapes: List[Tuple[int, int]]) -> List[np.ndarray]:
        """
        Reconstructs the original nested structure from flattened predictions.
        """
        entailment_matrices = []
        for i, shape in enumerate(shapes):
            entail_matrix = np.zeros(shape, dtype=int)
            for pred, (idx, j, k) in zip(flat_predictions, indices):
                if idx == i:
                    entail_matrix[j, k] = pred
            entailment_matrices.append(entail_matrix)

        return entailment_matrices


# from pydantic import BaseModel, Field
# from typing import Any, Optional, Literal, List, Tuple, Union
# class NLIResult(BaseModel):
#     """
#     Result from NLI prediction with probabilities.

#     This unified model supports both binary and ternary NLI styles.
#     The structure adapts based on the `style` field.
#     """

#     style: Literal["binary", "ternary"] = Field(..., description="The NLI style used")

#     # Binary fields (populated when style="binary")
#     binary_label: Optional[bool] = Field(None, description="True if entailed, False otherwise (binary style only)")
#     binary_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Probability of entailment (binary style only)")

#     # Ternary fields (populated when style="ternary")
#     ternary_label: Optional[Literal["contradiction", "neutral", "entailment"]] = Field(None, description="Predicted NLI class (ternary style only)")
#     ternary_probabilities: Optional[Tuple[float, float, float]] = Field(None, description="Probabilities for [contradiction, neutral, entailment] (ternary style only)")

#     @property
#     def label(self) -> Union[bool, str]:
#         """Get the label regardless of style."""
#         if self.style == "binary":
#             return self.binary_label
#         else:  # ternary
#             return self.ternary_label

#     @property
#     def entailment_probability(self) -> Optional[float]:
#         """Get entailment probability regardless of style."""
#         if self.style == "binary" and self.binary_probability:
#             return self.binary_probability
#         elif self.style == "ternary" and self.ternary_probabilities:
#             return self.ternary_probabilities[2]
#         return None

#     @property
#     def contradiction_probability(self) -> Optional[float]:
#         """Get contradiction probability (ternary only)."""
#         if self.style == "ternary" and self.ternary_probabilities:
#             return self.ternary_probabilities[0]
#         return None

#     @property
#     def neutral_probability(self) -> Optional[float]:
#         """Get neutral probability (ternary only)."""
#         if self.style == "ternary" and self.ternary_probabilities:
#             return self.ternary_probabilities[1]
#         return None
