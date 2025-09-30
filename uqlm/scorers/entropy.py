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


from typing import Any, List, Optional, Union
import warnings
from langchain_core.messages import BaseMessage

from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier
from uqlm.utils.results import UQResult
import time


class SemanticEntropy(UncertaintyQuantifier):
    def __init__(
        self,
        llm=None,
        postprocessor: Any = None,
        device: Any = None,
        use_best: bool = True,
        best_response_selection: str = "discrete",
        system_prompt: Optional[str] = None,
        max_calls_per_min: Optional[int] = None,
        use_n_param: bool = False,
        sampling_temperature: float = 1.0,
        verbose: bool = False,
        nli_model_name: str = "microsoft/deberta-large-mnli",
        max_length: int = 2000,
        return_responses: str = "all",
    ) -> None:
        """
        Class for computing discrete and token-probability-based semantic entropy and associated confidence scores. For more on semantic entropy, refer to Farquhar et al.(2024) :footcite:`farquhar2024detectinghallucinations`.

        Parameters
        ----------
        llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        postprocessor : callable, default=None
            A user-defined function that takes a string input and returns a string. Used for postprocessing
            outputs before black-box comparisons.

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. Only applies to 'semantic_negentropy', 'noncontradiction'
            scorers. Pass a torch.device to leverage GPU.

        use_best : bool, default=True
            Specifies whether to swap the original response for the uncertainty-minimized response
            based on semantic entropy clusters.

        best_response_selection : str, default="discrete"
            Specifies the type of entropy confidence score to compute best response. Must be one of "discrete" or "token-based".

        system_prompt : str, default=None
            Optional argument for user to provide custom system prompt. If prompts are list of strings and system_prompt is None,
            defaults to "You are a helpful assistant."

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.

        sampling_temperature : float, default=1.0
            The 'temperature' parameter for llm model to generate sampled LLM responses. Must be greater than 0.

        use_n_param : bool, default=False
            Specifies whether to use `n` parameter for `BaseChatModel`. Not compatible with all
            `BaseChatModel` classes. If used, it speeds up the generation process substantially when num_responses > 1.

        verbose : bool, default=False
            Specifies whether to print the index of response currently being scored.

        return_responses : str, default="all"
            If a postprocessor is used, specifies whether to return only postprocessed responses, only raw responses,
            or both. Specified with 'postprocessed', 'raw', or 'all', respectively.

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError
        """
        super().__init__(llm=llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param, postprocessor=postprocessor)
        self.nli_model_name = nli_model_name
        self.max_length = max_length
        self.verbose = verbose
        self.use_best = use_best
        self.sampling_temperature = sampling_temperature
        self.best_response_selection = best_response_selection
        self.return_responses = return_responses
        self._setup_nli(nli_model_name)
        self.prompts = None
        self.logprobs = None
        self.multiple_logprobs = None

    async def generate_and_score(self, prompts: List[Union[str, List[BaseMessage]]], num_responses: int = 5, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Evaluate discrete semantic entropy score on LLM responses for the provided prompts.

        Parameters
        ----------
        prompts : List[Union[str, List[BaseMessage]]]
            List of prompts from which LLM responses will be generated. Prompts in list may be strings or lists of BaseMessage. If providing
            input type List[List[BaseMessage]], refer to https://python.langchain.com/docs/concepts/messages/#langchain-messages for support.

        num_responses : int, default=5
            The number of sampled responses used to compute consistency.

        show_progress_bars : bool, default=True
            If True, displays a progress bar while generating and scoring responses

        Returns
        -------
        UQResult
            UQResult, containing data (prompts, responses, and semantic entropy scores) and metadata
        """
        self.prompts = prompts
        self.num_responses = num_responses
        self.nli_scorer.num_responses = num_responses

        if hasattr(self.llm, "logprobs"):
            self.llm.logprobs = True
        else:
            warnings.warn("The provided LLM does not support logprobs access. Only discrete semantic entropy will be computed.")

        self._construct_progress_bar(show_progress_bars)
        self._display_generation_header(show_progress_bars)

        responses = await self.generate_original_responses(prompts, progress_bar=self.progress_bar)
        sampled_responses = await self.generate_candidate_responses(prompts, num_responses=self.num_responses, progress_bar=self.progress_bar)
        return self.score(responses=responses, sampled_responses=sampled_responses, show_progress_bars=show_progress_bars)

    def score(self, responses: List[str] = None, sampled_responses: List[List[str]] = None, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Evaluate discrete semantic entropy score on LLM responses for the provided prompts.

        Parameters
        ----------
        responses : list of str, default=None
            A list of model responses for the prompts. If not provided, responses will be generated with the provided LLM.

        sampled_responses : list of list of str, default=None
            A list of lists of sampled model responses for each prompt. These will be used to compute consistency scores by comparing to
            the corresponding response from `responses`. If not provided, sampled_responses will be generated with the provided LLM.

        show_progress_bars : bool, default=True
            If True, displays a progress bar while scoring responses

        Returns
        -------
        UQResult
            UQResult, containing data (responses, sampled responses, and semantic entropy scores) and metadata
        """
        self.responses = responses
        self.sampled_responses = sampled_responses
        self.num_responses = len(self.sampled_responses[0])
        self.nli_scorer.num_responses = self.num_responses

        n_prompts = len(self.responses)
        discrete_semantic_entropy = [None] * n_prompts
        best_responses = [None] * n_prompts
        tokenprob_semantic_entropy = [None] * n_prompts

        def _process_i(i):
            candidates = [self.responses[i]] + self.sampled_responses[i]
            candidate_logprobs = [self.logprobs[i]] + self.multiple_logprobs[i] if (self.logprobs and self.multiple_logprobs) else None
            tmp = self.nli_scorer._semantic_entropy_process(candidates=candidates, i=i, logprobs_results=candidate_logprobs, best_response_selection=self.best_response_selection)
            best_responses[i], discrete_semantic_entropy[i], _, tokenprob_semantic_entropy[i] = tmp

        self._construct_progress_bar(show_progress_bars)
        self._display_scoring_header(show_progress_bars)
        if self.progress_bar:
            progress_task = self.progress_bar.add_task("- Scoring responses with NLI...", total=n_prompts)

        for i in range(n_prompts):
            _process_i(i)
            if self.progress_bar:
                self.progress_bar.update(progress_task, advance=1)
        time.sleep(0.1)
        confidence_scores = [1 - ne for ne in self.nli_scorer._normalize_entropy(discrete_semantic_entropy)]

        if self.use_best:
            self._update_best(best_responses, include_logprobs=self.llm.logprobs)

        data_to_return = self._construct_black_box_return_data()
        data_to_return["discrete_entropy_values"] = discrete_semantic_entropy
        data_to_return["discrete_confidence_scores"] = confidence_scores
        if tokenprob_semantic_entropy[0] is not None:
            data_to_return["tokenprob_entropy_values"] = tokenprob_semantic_entropy
            data_to_return["tokenprob_confidence_scores"] = [1 - ne for ne in self.nli_scorer._normalize_entropy(tokenprob_semantic_entropy)]

        result = {"data": data_to_return, "metadata": {"parameters": {"temperature": None if not self.llm else self.llm.temperature, "sampling_temperature": None if not self.sampling_temperature else self.sampling_temperature, "num_responses": self.num_responses}}}

        self._stop_progress_bar()
        self.progress_bar = None  # if re-run ensure the same progress object is not used
        return UQResult(result)
