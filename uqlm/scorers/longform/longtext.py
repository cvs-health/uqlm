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
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.scorers.longform.baseclass.uncertainty import LongFormUQ
from uqlm.utils.results import UQResult
from uqlm.longform.luq.matched_unit import MatchedUnitScorer
from uqlm.longform.luq.unit_response import UnitResponseScorer

UNIT_RESPONSE_SCORERS = ["entailment", "noncontradiction", "contrasted_entailment"]
MATCHED_UNIT_SCORERS = UNIT_RESPONSE_SCORERS + ["bert_score", "cosine_sim"]


class LongTextUQ(LongFormUQ):
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        granularity: str = "claim",
        mode: str = "unit_response",
        scorers: Optional[List[str]] = None,
        aggregation: str = "mean",
        response_refinement: bool = False,
        claim_filtering_scorer: Optional[str] = None,
        claim_decomposition_llm: Optional[BaseChatModel] = None,
        device: Any = None,
        nli_model_name: str = "microsoft/deberta-large-mnli",
        system_prompt: str = "You are a helpful assistant.",
        max_calls_per_min: Optional[int] = None,
        sampling_temperature: float = 1.0,
        use_n_param: bool = False,
        max_length: int = 2000,
    ) -> None:
        """
        Class for Long-text Uncertainty Quantification (LUQ) scorers.

        Parameters
        ----------
        llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        scorers : List[str], default=None
            Specifies which black box (consistency) scorers to include. List must be subset of ["entailment", "noncontradiction", "contrasted_entailment", "bert_score", "cosine_sim"].
            If None, defaults to ["entailment"].

        granularity : str, default="claim"
            Specifies whether to decompose and score at claim or sentence level granularity. Must be either "claim" or "sentence"

        aggregation : str, default="mean"
            Specifies how to aggregate claim/sentence-level scores to response-level scores. Must be one of 'min' or 'mean'.

        mode : str, default="unit_response"
            Specifies whether to implement unit-response (LUQ-style) scoring or matched-unit (LUQ-pair-style) scoring. Must be
            either "unit_response" or "matched_unit".

        response_refinement : bool, default=False
            Specifies whether to refine responses with uncertainty-aware decoding. This approach removes claims with confidence
            scores below the response_refinement_threshold and uses the claim_decomposition_llm to reconstruct the response from
            the retained claims. Only available for claim-level granularity. For more details, refer to
            Jiang et al., 2024: https://arxiv.org/abs/2410.20783

        claim_filtering_scorer : Optional[str], default=None
            specifies which scorer to use to filter claims if response_refinement is True. If not provided, defaults to the first
            element of self.scorers.

        claim_decomposition_llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel` to be used for decomposing responses into individual claims. Also used for claim refinement.
            If granularity="claim" and claim_decomposition_llm is None, the provided `llm` will be used for claim decomposition.

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. If None, detects and returns the best available PyTorch device.
            Prioritizes CUDA (NVIDIA GPU), then MPS (macOS), then CPU.

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        system_prompt : str or None, default="You are a helpful assistant."
            Optional argument for user to provide custom system prompt

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.

        sampling_temperature : float, default=1.0
            The 'temperature' parameter for llm model to generate sampled LLM responses. Must be greater than 0.

        use_n_param : bool, default=False
            Specifies whether to use `n` parameter for `BaseChatModel`. Not compatible with all
            `BaseChatModel` classes. If used, it speeds up the generation process substantially when num_responses > 1.

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError
        """
        self.scorers = ["entailment"] if not scorers else scorers
        super().__init__(llm=llm, granularity=granularity, aggregation=aggregation, scorers=self.scorers, response_refinement=response_refinement, claim_filtering_scorer=claim_filtering_scorer, claim_decomposition_llm=claim_decomposition_llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param)
        self.nli_model_name = nli_model_name
        self.mode = mode
        self.max_length = max_length
        self.sampling_temperature = sampling_temperature
        self._validate_scorers()
        self.prompts = None
        self.responses = None
        self.claim_sets = None
        self.sampled_responses = None
        self.sampled_claim_sets = None
        self.num_responses = None
        self.uad_result = {}

    async def generate_and_score(self, prompts: List[str], num_responses: int = 5, response_refinement_threshold: float = 1 / 3, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Generate LLM responses, sampled LLM (candidate) responses, and compute confidence scores with specified scorers for the provided prompts.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        num_responses : int, default=5
            The number of sampled responses used to compute consistency.

        response_refinement_threshold : float, default=1/3
            Threshold for uncertainty-aware filtering. Claims with confidence scores below this threshold are dropped from the
            refined response. Only used if response_refinement is True.

        show_progress_bars : bool, default=True
            If True, displays progress bars while generating and scoring responses

        Returns
        -------
        UQResult
            UQResult containing data (prompts, responses, and scores) and metadata
        """
        self.prompts = prompts
        self.num_responses = num_responses

        self._construct_progress_bar(show_progress_bars)
        self._display_generation_header(show_progress_bars)

        responses = await self.generate_original_responses(prompts=prompts, progress_bar=self.progress_bar)
        sampled_responses = await self.generate_candidate_responses(prompts=prompts, progress_bar=self.progress_bar, num_responses=self.num_responses)
        result = await self.score(responses=responses, sampled_responses=sampled_responses, response_refinement_threshold=response_refinement_threshold, show_progress_bars=show_progress_bars)
        return result

    async def score(self, responses: List[str], sampled_responses: List[List[str]], response_refinement_threshold: float = 1 / 3, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Compute confidence scores with specified scorers on provided LLM responses. Should only be used if responses and sampled responses
        are already generated. Otherwise, use `generate_and_score`.

        Parameters
        ----------
        responses : list of str, default=None
            A list of model responses for the prompts.

        sampled_responses : list of list of str, default=None
            A list of lists of sampled LLM responses for each prompt. These will be used to compute consistency scores by comparing to
            the corresponding response from `responses`.

        response_refinement_threshold : float, default=1/3
            Threshold for uncertainty-aware filtering. Claims with confidence scores below this threshold are dropped from the
            refined response. Only used if response_refinement is True.

        show_progress_bars : bool, default=True
            If True, displays a progress bar while scoring responses

        Returns
        -------
        UQResult
            UQResult containing data (responses and scores) and metadata
        """
        self.responses = responses
        self.sampled_responses = sampled_responses
        self.num_responses = len(sampled_responses[0])
        self._construct_progress_bar(show_progress_bars)

        await self._decompose_responses(show_progress_bars)
        if self.mode == "matched_unit":
            await self._decompose_candidate_responses(show_progress_bars)
        self._display_scoring_header(show_progress_bars)

        self.scores_dict = self._score_from_decomposed(claim_sets=self.claim_sets, sampled_responses=self.sampled_responses, sampled_claim_sets=self.sampled_claim_sets, progress_bar=self.progress_bar)

        if self.response_refinement:
            self.uad_result = await self.uncertainty_aware_decode(claim_sets=self.claim_sets, claim_scores=self.claim_scores[self.uad_scorer], response_refinement_threshold=response_refinement_threshold, show_progress_bars=show_progress_bars)
        self._stop_progress_bar()
        self.progress_bar = None

        claims_data = []
        for i in range(len(self.claim_sets)):
            claim_i_data = []
            for j in range(len(self.claim_sets[i])):
                claims_dict = {self.granularity: self.claim_sets[i][j], "removed": False if not self.uad_result else self.uad_result["removed"][i][j]}
                for scorer in self.scorers:
                    claims_dict[scorer] = self.claim_scores[scorer][i][j]
                claim_i_data.append(claims_dict)
            claims_data.append(claim_i_data)

        self.scores_dict["claims_data"] = claims_data
        if "removed" in self.uad_result:
            del self.uad_result["removed"]

        return self._construct_result()

    def _score_from_decomposed(self, claim_sets: List[List[str]], sampled_responses: Optional[List[List[str]]] = None, sampled_claim_sets: Optional[List[List[List[str]]]] = None, progress_bar: Optional[Progress] = None) -> UQResult:
        """
        Compute confidence scores with specified scorers on provided LLM responses. Should only be used if responses and sampled responses
        are already generated. Otherwise, use `generate_and_score`.

        Parameters
        ----------
        claim_sets : list of list of strings
            List of original responses decomposed into lists of either claims or sentences

        sampled_responses : list of list of strings
            Candidate responses to be compared to the decomposed original responses

        sampled_claim_sets : list of list of list of strings
            Decomposed responses to be compared to the decomposed original responses

        Returns
        -------
        UQResult
            UQResult containing data (responses and scores) and metadata
        """
        if self.mode == "unit_response":
            self.claim_scores = self.unit_response_scorer.evaluate(claim_sets=self.claim_sets, sampled_responses=sampled_responses, progress_bar=progress_bar).to_dict()
        elif self.mode == "matched_unit":
            self.claim_scores = self.matched_unit_scorer.evaluate(claim_sets=self.claim_sets, sampled_claim_sets=self.sampled_claim_sets, progress_bar=progress_bar).to_dict()

        scores_dict = {}
        for scorer in self.scorers:
            scores_dict[scorer] = self._aggregate_scores(self.claim_scores[scorer])

        return scores_dict

    def _construct_result(self) -> Any:
        """Constructs UQResult object"""
        data = {"responses": self.responses, "sampled_responses": self.sampled_responses}
        if self.prompts:
            data["prompts"] = self.prompts
        # if self.claim_sets:
        #     data[self.granularity + "s"] = self.claim_sets
        data.update(self.scores_dict)
        data.update(self.uad_result)
        result = {"data": data, "metadata": {"mode": self.mode, "granularity": self.granularity, "aggregation": self.aggregation, "temperature": None if not self.llm else self.llm.temperature, "sampling_temperature": None if not self.sampling_temperature else self.sampling_temperature, "num_responses": self.num_responses, "response_refinement_threshold": self.response_refinement_threshold}}
        return UQResult(result)

    def _validate_scorers(self) -> None:
        """Validate scorers"""
        self.matched_unit_scorer = None
        self.unit_response_scorer = None
        if self.mode == "unit_response":
            if set(self.scorers) - set(UNIT_RESPONSE_SCORERS):
                raise ValueError(
                    f"""
                Invalid scorers: {set(self.scorers) - set(UNIT_RESPONSE_SCORERS)}. Must be subset of {UNIT_RESPONSE_SCORERS} when mode="unit_response"
                """
                )
            self.unit_response_scorer = UnitResponseScorer(nli_model_name=self.nli_model_name, device=self.device, max_length=self.max_length)
        elif self.mode == "matched_unit":
            self.matched_unit_scorer = MatchedUnitScorer(nli_model_name=self.nli_model_name, device=self.device, max_length=self.max_length)
            if set(self.scorers) - set(MATCHED_UNIT_SCORERS):
                raise ValueError(
                    f"""
                Invalid scorers: {set(self.scorers) - set(MATCHED_UNIT_SCORERS)}. Must be subset of {MATCHED_UNIT_SCORERS} when mode="matched_unit"
                """
                )
        else:
            raise ValueError(
                f"""
                Invalid mode: {self.mode}. Must be one of "unit_response", "matched_unit"
                """
            )
