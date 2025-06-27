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

from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier, UQResult


class SemanticEntropy(UncertaintyQuantifier):
    def __init__(self, llm=None, postprocessor: Any = None, device: Any = None, use_best: bool = True, system_prompt: str = "You are a helpful assistant.", max_calls_per_min: Optional[int] = None, use_n_param: bool = False, sampling_temperature: float = 1.0, verbose: bool = False, nli_model_name: str = "microsoft/deberta-large-mnli", max_length: int = 2000, discrete: bool = True) -> None:
        """
        Class for computing Discrete Semantic Entropy-based confidence scores. For more on semantic entropy,
        refer to Farquhar et al.(2024) :footcite:`farquhar2024detectinghallucinations`.

        Parameters
        ----------
        llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        postprocessor : callable, default=None
            A user-defined function that takes a string input and returns a string. Used for postprocessing
            outputs.

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. Only applies to 'semantic_negentropy', 'noncontradiction'
            scorers. Pass a torch.device to leverage GPU.

        use_best : bool, default=True
            Specifies whether to swap the original response for the uncertainty-minimized response
            based on semantic entropy clusters.

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
        self.prompts = None
        self._setup_nli(nli_model_name)
        self.nli_scorer.discrete = discrete

    async def generate_and_score(self, prompts: List[str], num_responses: int = 5) -> UQResult:
        """
        Evaluate discrete semantic entropy score on LLM responses for the provided prompts.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        num_responses : int, default=5
            The number of sampled responses used to compute consistency.

        Returns
        -------
        UQResult
            UQResult, containing data (prompts, responses, and semantic entropy scores) and metadata
        """
        self.prompts = prompts
        self.num_responses = num_responses
        self.nli_scorer.num_responses = num_responses

        responses = await self.generate_original_responses(prompts)
        sampled_responses = await self.generate_candidate_responses(prompts)
        return self.score(responses=responses, sampled_responses=sampled_responses)

    def score(self, responses: List[str] = None, sampled_responses: List[List[str]] = None) -> UQResult:
        """
        Evaluate discrete semantic entropy score on LLM responses for the provided prompts.

        Parameters
        ----------
        responses : list of str, default=None
            A list of model responses for the prompts. If not provided, responses will be generated with the provided LLM.

        sampled_responses : list of list of str, default=None
            A list of lists of sampled model responses for each prompt. These will be used to compute consistency scores by comparing to
            the corresponding response from `responses`. If not provided, sampled_responses will be generated with the provided LLM.

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
        semantic_entropy = [None] * n_prompts
        best_responses = [None] * n_prompts

        print("Computing confidence scores...")
        for i in range(n_prompts):
            candidates = [self.responses[i]] + self.sampled_responses[i]
            tmp = self.nli_scorer._semantic_entropy_process(candidates=candidates, i=i)
            best_responses[i], semantic_entropy[i], scores = tmp

        confidence_scores = [1 - ne for ne in self.nli_scorer._normalize_entropy(semantic_entropy)]

        result = {
            "data": {"responses": best_responses if self.use_best else self.responses, "entropy_values": semantic_entropy, "confidence_scores": confidence_scores, "sampled_responses": self.sampled_responses},
            "metadata": {"parameters": {"temperature": None if not self.llm else self.llm.temperature, "sampling_temperature": None if not self.sampling_temperature else self.sampling_temperature, "num_responses": self.num_responses}},
        }
        if self.prompts:
            result["data"]["prompts"] = self.prompts
        return UQResult(result)
