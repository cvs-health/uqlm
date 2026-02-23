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

from typing import Any, Dict, List, Optional, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from uqlm.white_box.single_logprobs import SingleLogprobsScorer, SINGLE_LOGPROBS_SCORER_NAMES
from uqlm.white_box.top_logprobs import TopLogprobsScorer, TOP_LOGPROBS_SCORER_NAMES
from uqlm.white_box.sampled_logprobs import SampledLogprobsScorer, SAMPLED_LOGPROBS_SCORER_NAMES
from uqlm.white_box.p_true import PTrueScorer
from uqlm.scorers.shortform.baseclass.uncertainty import ShortFormUQ
from uqlm.utils.results import UQResult
from uqlm.utils.warn import beta_warning

ALL_WHITE_BOX_SCORER_NAMES = SINGLE_LOGPROBS_SCORER_NAMES + TOP_LOGPROBS_SCORER_NAMES + SAMPLED_LOGPROBS_SCORER_NAMES + ["p_true"]

SCORERS_FOR_SCORING_HEADER = ["consistency_and_confidence", "semantic_negentropy", "semantic_density", "p_true"]


class WhiteBoxUQ(ShortFormUQ):
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        system_prompt: Optional[str] = None,
        max_calls_per_min: Optional[int] = None,
        scorers: Optional[List[str]] = None,
        sampling_temperature: float = 1.0,
        top_k_logprobs: int = 15,
        use_n_param: bool = False,
        length_normalize: bool = True,
        prompts_in_nli: bool = True,
        device: Any = None,
        max_length: int = 2000,
        sentence_transformer: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model_name: str = "microsoft/deberta-large-mnli",
    ) -> None:
        """
        Class for computing white-box UQ confidence scores. This class offers two confidence scores, normalized
        probability :footcite:`malinin2021uncertaintyestimationautoregressivestructured` and minimum probability :footcite:`manakul2023selfcheckgptzeroresourceblackboxhallucination`.

        Parameters
        ----------
        llm : BaseChatModel
            A langchain llm object to get passed to chain constructor. User is responsible for specifying
            temperature and other relevant parameters to the constructor of their `llm` object.

        max_calls_per_min : int, default=None
            Used to control rate limiting.

        system_prompt : str, default=None
            Optional argument for user to provide custom system prompt. If prompts are list of strings and system_prompt is None,
            defaults to "You are a helpful assistant."

        scorers : List[str], default=None
            Specifies which white-box UQ scorers to include. Must be subset of ["sequence_probability", "min_probability", "max_token_negentropy", "mean_token_negentropy", "probability_margin", "monte_carlo_probability", "consistency_and_confidence", "semantic_negentropy", "semantic_density", "p_true"]. If None, defaults to ["sequence_probability", "min_probability"].

        sampling_temperature : float, default=1.0
            The 'temperature' parameter for llm model to generate sampled LLM responses. Must be greater than 0.

        use_n_param : bool, default=False
            Specifies whether to use `n` parameter for `BaseChatModel`. Not compatible with all
            `BaseChatModel` classes. If used, it speeds up the generation process substantially when num_responses > 1.

        prompts_in_nli : bool, default=True
            Specifies whether to use the prompts in the NLI inputs for semantic entropy and semantic density scorers.

        length_normalize : bool, default=True
            Specifies whether to length normalize the logprobs. This attribute affects the response probability computation for sequence_probability, semantic_negentropy, semantic_density, monte_carlo_probability, and consistency_and_confidence.

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. Only applies to 'semantic_negentropy', 'semantic_density' scorers. If None, detects and returns the best available PyTorch device.
            Prioritizes CUDA (NVIDIA GPU), then MPS (macOS), then CPU.

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError

        sentence_transformer : str, default="sentence-transformers/all-MiniLM-L6-v2"
            Specifies which huggingface sentence transformer to use when computing cosine similarity for consistency_and_confidence. See
            https://huggingface.co/sentence-transformers?sort_models=likes#models
            for more information. The recommended sentence transformer is 'sentence-transformers/all-MiniLM-L6-v2'.

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        """
        super().__init__(llm=llm, max_calls_per_min=max_calls_per_min, system_prompt=system_prompt)
        self.sampling_temperature = sampling_temperature
        self.top_k_logprobs = None  # used only if top_logprobs scorers used
        self.length_normalize = length_normalize
        self.prompts_in_nli = prompts_in_nli
        self.device = device
        self.max_length = max_length
        self.scorers_with_scoring_header = False
        self.sentence_transformer = sentence_transformer
        self.nli_model_name = nli_model_name
        self._validate_scorers(scorers, top_k_logprobs)
        self.multiple_logprobs = None

    async def generate_and_score(self, prompts: List[Union[str, List[BaseMessage]]], num_responses: Optional[int] = 5, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Generate responses and compute white-box confidence scores based on extracted token probabilities.

        Parameters
        ----------
        prompts : List[Union[str, List[BaseMessage]]]
            List of prompts from which LLM responses will be generated. Prompts in list may be strings or lists of BaseMessage. If providing
            input type List[List[BaseMessage]], refer to https://python.langchain.com/docs/concepts/messages/#langchain-messages for support.

        num_responses : int, default=5
            The number of sampled responses used to multi-generation white-box scorers. Only applies to "monte_carlo_probability", "consistency_and_confidence", "semantic_negentropy", "semantic_density" scorers.

        show_progress_bars : bool, default=True
            If True, displays a progress bar while generating and scoring responses

        Returns
        -------
        UQResult
            UQResult containing prompts, responses, logprobs, and white-box UQ scores
        """
        assert hasattr(self.llm, "logprobs"), """
        BaseChatModel must have logprobs attribute and have logprobs=True
        """
        self.llm.logprobs = True
        sampled_responses = None

        self._construct_progress_bar(show_progress_bars)
        self._display_generation_header(show_progress_bars, generation_type="white_box")

        responses = await self.generate_original_responses(prompts, top_k_logprobs=self.top_k_logprobs, progress_bar=self.progress_bar)
        if self.sampled_logprobs_scorer_names:
            self.llm.logprobs = True  # reset attribute to True
            sampled_responses = await self.generate_candidate_responses(prompts=prompts, num_responses=num_responses, progress_bar=self.progress_bar)
        result = await self.score(prompts=prompts, responses=responses, sampled_responses=sampled_responses, logprobs_results=self.logprobs, sampled_logprobs_results=self.multiple_logprobs, show_progress_bars=show_progress_bars)

        self._stop_progress_bar()
        self.progress_bar = None  # if re-run ensure the same progress object is not used
        return result

    async def score(self, logprobs_results: List[List[Dict[str, Any]]], prompts: Optional[List[str]] = None, responses: Optional[List[str]] = None, sampled_responses: Optional[List[List[str]]] = None, sampled_logprobs_results: Optional[List[List[List[Dict[str, Any]]]]] = None, show_progress_bars: Optional[bool] = True, _display_header: bool = True) -> UQResult:
        """
        Compute white-box confidence scores from provided logprobs.

        Parameters
        ----------
        logprobs_results : list of logprobs_result
            List of dictionaries, each returned by BaseChatModel.ainvoke

        prompts : list of str, default=None
            A list of input prompts for the model. Required only for "p_true" scorer.

        responses : list of str, default=None
            A list of model responses for the prompts. Required for "monte_carlo_probability", "consistency_and_confidence", "semantic_negentropy", "semantic_density", "p_true" scorers.

        sampled_responses : list of list of str, default=None
            A list of lists of sampled LLM responses for each prompt. These will be used to compute consistency scores by comparing to
            the corresponding response from `responses`. Required for "monte_carlo_probability", "consistency_and_confidence", "semantic_negentropy", "semantic_density" scorers.

        sampled_logprobs_results : list of lists of logprobs_result
            List of list of dictionaries, each returned by BaseChatModel.ainvoke corresponding to sampled_responses. Required only for "monte_carlo_probability", "semantic_negentropy", "semantic_density" scorers.

        show_progress_bars : bool, default=True
            If True, displays a progress bar while scoring responses

        Returns
        -------
        UQResult
            UQResult containing prompts, responses, logprobs, and white-box UQ scores
        """
        self._construct_progress_bar(show_progress_bars)
        self._display_scoring_header(show_progress_bars and _display_header and self.scorers_with_scoring_header)

        data = {"prompts": prompts, "responses": responses, "logprob": logprobs_results, "sampled_responses": sampled_responses, "sampled_logprob": sampled_logprobs_results}
        data = {key: val for key, val in data.items() if val}

        if self.single_logprobs_scorer_names:
            single_logprobs_scores_dict = self.single_logprobs_scorer.evaluate(logprobs_results)
            data.update(single_logprobs_scores_dict)
        if self.top_logprobs_scorer_names:
            top_logprobs_scores_dict = self.top_logprobs_scorer.evaluate(logprobs_results)
            data.update(top_logprobs_scores_dict)
        if self.sampled_logprobs_scorer_names:
            sampled_logprobs_scores_dict = self.sampled_logprobs_scorer.evaluate(logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results, responses=responses, sampled_responses=sampled_responses, prompts=prompts, progress_bar=self.progress_bar)
            data.update(sampled_logprobs_scores_dict)

        self._start_progress_bar()  # restart progress bar as entropy scorer stops it
        if "p_true" in self.scorers:
            p_true_scores_dict = await self.p_true_scorer.evaluate(prompts=prompts, responses=responses, sampled_responses=sampled_responses, progress_bar=self.progress_bar)
            data.update(p_true_scores_dict)
        result = {"data": data, "metadata": {"temperature": None if not self.llm else self.llm.temperature}}
        return UQResult(result)

    def _validate_scorers(self, scorers: List[str], top_k_logprobs: int) -> None:
        """Validate and store scorer list"""
        if not scorers:
            self.scorers = self.white_box_names
        else:
            if "normalized_probability" in scorers:
                raise ValueError("normalized_probability is deprecated as of v0.5 in favor of sequence_probability with length_normalize=True")
            self.scorers = []
            for scorer in scorers:
                if scorer in ALL_WHITE_BOX_SCORER_NAMES:
                    self.scorers.append(scorer)
                else:
                    raise ValueError(f"Invalid scorer provided: {scorer}")
        self.single_logprobs_scorer_names = list(set(SINGLE_LOGPROBS_SCORER_NAMES) & set(self.scorers))
        self.top_logprobs_scorer_names = list(set(TOP_LOGPROBS_SCORER_NAMES) & set(self.scorers))
        self.sampled_logprobs_scorer_names = list(set(SAMPLED_LOGPROBS_SCORER_NAMES) & set(self.scorers))
        if self.single_logprobs_scorer_names:
            self.single_logprobs_scorer = SingleLogprobsScorer(scorers=self.single_logprobs_scorer_names, length_normalize=self.length_normalize)
        if self.top_logprobs_scorer_names:
            self.top_logprobs_scorer = TopLogprobsScorer(scorers=self.top_logprobs_scorer_names)
            self.top_k_logprobs = top_k_logprobs
            beta_warning("Scorers based on top_logprobs ('mean_token_negentropy','min_token_negentropy','probability_margin') is in beta. Please use with caution as it may change in future releases.")
        if self.sampled_logprobs_scorer_names:
            self.sampled_logprobs_scorer = SampledLogprobsScorer(scorers=self.sampled_logprobs_scorer_names, llm=self.llm, prompts_in_nli=self.prompts_in_nli, length_normalize=self.length_normalize, device=self.device, max_length=self.max_length, sentence_transformer=self.sentence_transformer, nli_model_name=self.nli_model_name)
        if "p_true" in self.scorers:
            self.p_true_scorer = PTrueScorer(llm=self.llm, max_calls_per_min=self.max_calls_per_min)
        if set(SCORERS_FOR_SCORING_HEADER) & set(self.scorers):
            self.scorers_with_scoring_header = True
