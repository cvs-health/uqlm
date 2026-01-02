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

from uqlm.scorers.shortform.baseclass.uncertainty import ShortFormUQ
from uqlm.utils.results import UQResult
from typing import Any, Optional, List
import time
from uqlm.nli.cluster import SemanticClusterer
from typing import Dict
import numpy as np


class SemanticDensity(ShortFormUQ):
    def __init__(self, llm=None, postprocessor: Any = None, device: Any = None, system_prompt: str = "You are a helpful assistant.", max_calls_per_min: Optional[int] = None, use_n_param: bool = False, sampling_temperature: float = 1.0, verbose: bool = False, nli_model_name: str = "microsoft/deberta-large-mnli", max_length: int = 2000, return_responses: str = "all", length_normalize: bool = True):
        """
        Class for computing semantic density and associated confidence scores. For more on semantic density, refer to Qiu et al.(2024) :footcite:`qiu2024semanticdensityuncertaintyquantification`.

        Parameters
        ----------
        llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        postprocessor : callable, default=None
            A user-defined function that takes a string input and returns a string. Used for postprocessing
            outputs before black-box comparisons.

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. If None, detects and returns the best available PyTorch device.
            Prioritizes CUDA (NVIDIA GPU), then MPS (macOS), then CPU.

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

        length_normalize : bool, default=True
            Determines whether response probabilities are length-normalized. Recommended to set as True when longer responses are expected.
        """
        super().__init__(llm=llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param, postprocessor=postprocessor)
        self.nli_model_name = nli_model_name
        self.max_length = max_length
        self.verbose = verbose
        self.sampling_temperature = sampling_temperature
        self.return_responses = return_responses
        self._setup_nli(nli_model_name)
        self.prompts = None
        self.clusterer = SemanticClusterer(nli=self.nli, length_normalize=length_normalize)

    async def generate_and_score(self, prompts: List[str], num_responses: int = 5, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Evaluate semantic density score on LLM responses for the provided prompts.

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
            UQResult, containing data (prompts, responses, and semantic density score) and metadata
        """
        self.prompts = prompts
        self.num_responses = num_responses
        self.nli.num_responses = num_responses

        if hasattr(self.llm, "logprobs"):
            self.llm.logprobs = True
        else:
            raise ValueError("The provided LLM does not support logprobs access. Cannot compute semantic density.")

        self._construct_progress_bar(show_progress_bars)
        self._display_generation_header(show_progress_bars)

        responses = await self.generate_original_responses(prompts, progress_bar=self.progress_bar)
        sampled_responses = await self.generate_candidate_responses(prompts, num_responses=self.num_responses, progress_bar=self.progress_bar)
        return self.score(prompts=self.prompts, responses=responses, sampled_responses=sampled_responses, show_progress_bars=show_progress_bars)

    def score(self, prompts: List[str] = None, responses: List[str] = None, sampled_responses: List[List[str]] = None, logprobs_results: List[List[Dict[str, Any]]] = None, sampled_logprobs_results: List[List[List[Dict[str, Any]]]] = None, show_progress_bars: Optional[bool] = True, _display_header: bool = True) -> UQResult:
        """
        Evaluate semantic density score on LLM responses for the provided prompts.

        Parameters
        ----------
        prompts : List[Union[str, List[BaseMessage]]]
            List of prompts from which LLM responses will be generated. Prompts in list may be strings or lists of BaseMessage. If providing
            input type List[List[BaseMessage]], refer to https://python.langchain.com/docs/concepts/messages/#langchain-messages for support.

        responses : list of str, default=None
            A list of model responses for the prompts. If not provided, responses will be generated with the provided LLM.

        sampled_responses : list of list of str, default=None
            A list of lists of sampled model responses for each prompt. These will be used to compute consistency scores by comparing to
            the corresponding response from `responses`. If not provided, sampled_responses will be generated with the provided LLM.

        logprobs_results : list of list of dict, default=None
            A list of lists of logprobs results for each prompt. If not provided, logprobs will be generated with the provided LLM.

        sampled_logprobs_results : list of list of list of dict, default=None
            A list of lists of lists of logprobs results for each prompt. If not provided, sampled_logprobs will be generated with the provided LLM.

        show_progress_bars : bool, default=True
            If True, displays a progress bar while scoring responses

        Returns
        -------
        UQResult
            UQResult, containing data (responses, sampled responses, and semantic density score) and metadata
        """
        self.prompts = prompts if prompts else self.prompts
        self.responses = responses
        self.sampled_responses = sampled_responses
        self.num_responses = len(self.sampled_responses[0])
        self.nli.num_responses = self.num_responses
        self.logprobs = logprobs_results if logprobs_results else self.logprobs
        self.multiple_logprobs = sampled_logprobs_results if sampled_logprobs_results else self.multiple_logprobs

        n_prompts = len(self.responses)
        semantic_density = [None] * n_prompts

        def _process_i(i):
            prompt = self.prompts[i]

            original_response = self.responses[i]

            candidates = self.sampled_responses[i]
            candidate_logprobs = self.multiple_logprobs[i]
            semantic_density[i], _ = self._semantic_density_process(prompt=prompt, original_response=original_response, candidates=candidates, i=i, logprobs_results=candidate_logprobs)

        self._construct_progress_bar(show_progress_bars)
        self._display_scoring_header(show_progress_bars and _display_header)
        if self.progress_bar:
            progress_task = self.progress_bar.add_task("  - Scoring responses with semantic clustering...", total=n_prompts)

        for i in range(n_prompts):
            _process_i(i)
            if self.progress_bar:
                self.progress_bar.update(progress_task, advance=1)
        time.sleep(0.1)

        data_to_return = self._construct_black_box_return_data()
        data_to_return["semantic_density_values"] = semantic_density
        data_to_return["multiple_logprobs"] = self.multiple_logprobs

        result = {"data": data_to_return, "metadata": {"parameters": {"temperature": None if not self.llm else self.llm.temperature, "sampling_temperature": None if not self.sampling_temperature else self.sampling_temperature, "num_responses": self.num_responses}}}

        self._stop_progress_bar()
        self.progress_bar = None  # if re-run ensure the same progress object is not used
        return UQResult(result)

    def _semantic_density_process(self, prompt: str, original_response: str, candidates: List[str], i: int = None, logprobs_results: List[List[Dict[str, Any]]] = None) -> Any:
        """
        Executes complete process for semantic density and returns SD score, and dictionary
        of NLI scores for response pairs
        """
        if self.verbose and i is not None:
            print("Question No. - ", i + 1)

        # Get the length-normalized tokenwise probability for each candidate response
        tokenprob_response_probabilities, _ = self.clusterer.compute_response_probabilities(logprobs_results=logprobs_results, num_responses=len(candidates))

        # Compute entailment of each candidate response by the original response,
        # conditioned on prompt
        nli_scores = []
        for candidate in candidates:
            inputs = (f"{prompt}\n{original_response}", f"{prompt}\n{candidate}")
            if inputs[0] + "_" + inputs[1] not in self.nli.probabilities:
                nli_scores.append(self.nli.predict(inputs[0], inputs[1]))
            else:
                nli_scores.append(self.nli.probabilities[inputs[0] + "_" + inputs[1]])

        # Use NLI model to estimate semantic distance between each candidate response
        # and the original response
        contradiction_index, neutral_index = (self.nli.label_mapping.index("contradiction"), self.nli.label_mapping.index("neutral"))

        semantic_distance_expectation = np.array([nli_score[0, contradiction_index] + nli_score[0, neutral_index] * (np.sqrt(2) / 2) for nli_score in nli_scores])
        semantic_squared_distance_expectation = np.array([nli_score[0, contradiction_index] + nli_score[0, neutral_index] / 2 for nli_score in nli_scores])

        # Evaluate the kernel function for each candidate response
        indicator = semantic_distance_expectation <= 1
        kernel_values = (1 - semantic_squared_distance_expectation) * indicator

        # Calculate final semantic density score
        if np.sum(tokenprob_response_probabilities) == 0:
            semantic_density = np.nan
        else:
            semantic_density = np.average(kernel_values, weights=tokenprob_response_probabilities)

        return (semantic_density, nli_scores)
