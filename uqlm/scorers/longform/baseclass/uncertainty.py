# Copyright 2025 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, List, Optional
import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier
from uqlm.longform.decomposition import ResponseDecomposer
from uqlm.longform.uad import UncertaintyAwareDecoder


class LongFormUQ(UncertaintyQuantifier):
    def __init__(
        self, llm: Any = None, scorers: Optional[List[str]] = None, granularity: str = "claim", aggregation: str = "mean", claim_decomposition_llm: Optional[BaseChatModel] = None, response_refinement: bool = False, claim_filtering_scorer: Optional[str] = None, device: Any = None, system_prompt: Optional[str] = None, max_calls_per_min: Optional[int] = None, use_n_param: bool = False
    ) -> None:
        """
        Parent class for uncertainty quantification of LLM responses

        Parameters
        ----------
        llm : BaseChatModel
            A langchain llm object to get passed to chain constructor. User is responsible for specifying
            temperature and other relevant parameters to the constructor of their `llm` object.

        scorers : List[str], default=None
            Specifies which black box (consistency) scorers to include.

        aggregation : str, default="mean"
            Specifies how to aggregate claim/sentence-level scores to response-level scores. Must be one of 'min' or 'mean'.

        granularity : str, default="claim"
            Specifies whether to decompose and score at claim or sentence level granularity. Must be either "claim" or "sentence"

        claim_decomposition_llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel` to be used for decomposing responses into individual claims. Also used for claim refinement.
            If granularity="claim" and claim_decomposition_llm is None, the provided `llm` will be used for claim decomposition.

        response_refinement : bool, default=False
            Specifies whether to refine responses with uncertainty-aware decoding. This approach removes claims with confidence
            scores below the response_refinement_threshold and uses the claim_decomposition_llm to reconstruct the response from
            the retained claims. Only available for claim-level granularity. For more details, refer to
            Jiang et al., 2024: https://arxiv.org/abs/2410.20783

        claim_filtering_scorer : Optional[str], default=None
            specifies which scorer to use to filter claims if response_refinement is True. If not provided, defaults to the first
            element of self.scorers.

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. Only applies to 'semantic_negentropy', 'noncontradiction'
            scorers. Pass a torch.device to leverage GPU.

        system_prompt : str, default=None
            Optional argument for user to provide custom system prompt. If prompts are list of strings and system_prompt is None,
            defaults to "You are a helpful assistant."

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.

        use_n_param : bool, default=False
            Specifies whether to use `n` parameter for `BaseChatModel`. Not compatible with all
            `BaseChatModel` classes. If used, it speeds up the generation process substantially when num_responses > 1.
        """
        super().__init__(llm=llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param)
        self.claim_decomposition_llm = claim_decomposition_llm
        self.decomposer = ResponseDecomposer(claim_decomposition_llm=claim_decomposition_llm if claim_decomposition_llm else llm)
        self.granularity = granularity
        self.scorers = scorers
        self.aggregation = aggregation
        self.response_refinement = response_refinement
        self.claim_filtering_scorer = claim_filtering_scorer
        self.response_refinement_threshold = None
        if self.granularity not in ["sentence", "claim"]:
            raise ValueError(
                f"""
                Invalid granularity: {self.granularity}. Must be one of "sentence", "claim"
                """
            )
        if self.response_refinement:
            if self.granularity != "claim":
                raise ValueError("Uncertainty aware decoding is only possible with claim-level scoring. Please set response_refinement=False or granularity='claim'")
            self.reconstructor = UncertaintyAwareDecoder(reconstructor_llm=self.decomposer.claim_decomposition_llm)
            if not self.claim_filtering_scorer:
                print(f"claim_filtering_scorer is not specified for response_refinement. Defaulting to {self.scorers[0]}.")
                self.uad_scorer = self.scorers[0]
            elif self.claim_filtering_scorer not in self.scorers:
                print(f"claim_filtering_scorer is contained in list of scorers. Defaulting to {self.scorers[0]}.")
                self.uad_scorer = self.scorers[0]
            else:
                self.uad_scorer = self.claim_filtering_scorer

    async def uncertainty_aware_decode(self, claim_sets: List[List[str]], claim_scores: List[List[float]], response_refinement_threshold: float = 1 / 3, show_progress_bars: Optional[bool] = True) -> List[str]:
        """
        Parameters
        ----------
        claim_sets : List[List[str]]
            List of original responses decomposed into lists of claims

        claim_scores : List[List[float]]
            List of lists of claim-level confidence scores to be used for uncertainty-aware filtering

        response_refinement_threshold : float, default=1/3
            Threshold for uncertainty-aware filtering. Claims with confidence scores below this threshold are dropped from the
            refined response. Only used if response_refinement is True.

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses
        """
        self._construct_progress_bar(show_progress_bars)
        self._display_reconstruction_header(show_progress_bars)
        self.response_refinement_threshold = response_refinement_threshold
        uad_result = await self.reconstructor.reconstruct_responses(claim_sets=claim_sets, claim_scores=claim_scores, responses=self.responses, threshold=response_refinement_threshold, progress_bar=self.progress_bar)
        self._stop_progress_bar()
        self.progress_bar = None

        for scorer in self.scorers:
            filtered_claim_scores = []
            for i in range(len(claim_sets)):
                filtered_claim_scores_i = []
                for j in range(len(claim_sets[i])):
                    if not uad_result["removed"][i][j]:
                        filtered_claim_scores_i.append(self.claim_scores[scorer][i][j])
                filtered_claim_scores.append(filtered_claim_scores_i)

            uad_result["refined_" + scorer] = self._aggregate_scores(filtered_claim_scores)

        return uad_result

    async def _decompose_responses(self, show_progress_bars) -> None:
        """Decompose original responses into claims or sentences"""
        self._display_decomposition_header(show_progress_bars)
        if self.granularity == "sentence":
            self.claim_sets = self.decomposer.decompose_sentences(responses=self.responses, progress_bar=self.progress_bar)
        elif self.granularity == "claim":
            self.claim_sets = await self.decomposer.decompose_claims(responses=self.responses, progress_bar=self.progress_bar)

    async def _decompose_candidate_responses(self, show_progress_bars) -> None:
        """Display header and decompose responses"""
        if self.granularity == "sentence":
            self.sampled_claim_sets = self.decomposer.decompose_candidate_sentences(sampled_responses=self.sampled_responses, progress_bar=self.progress_bar)
        elif self.granularity == "claim":
            self.sampled_claim_sets = await self.decomposer.decompose_candidate_claims(sampled_responses=self.sampled_responses, progress_bar=self.progress_bar)

    def _aggregate_scores(self, claim_scores: List[List[float]]) -> List[float]:
        """Aggregate claim scores to response level scores"""
        if self.aggregation == "mean":
            return [np.mean(cs) for cs in claim_scores]
        elif self.aggregation == "min":
            return [np.min(cs) for cs in claim_scores]

    def _display_decomposition_header(self, show_progress_bars: bool) -> None:
        """Displays decomposition header"""
        if show_progress_bars:
            self.progress_bar.start()
            self.progress_bar.add_task("")
            self.progress_bar.add_task("✂️ Decomposition")

    def _display_reconstruction_header(self, show_progress_bars: bool) -> None:
        """Displays decomposition header"""
        if show_progress_bars:
            self.progress_bar.start()
            self.progress_bar.add_task("")
            self.progress_bar.add_task("✅️ Refinement")
