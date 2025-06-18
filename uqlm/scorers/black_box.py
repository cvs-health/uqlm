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


from langchain_core.language_models.chat_models import BaseChatModel
from typing import Any, List, Optional

from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier, UQResult
from uqlm.black_box import BertScorer, CosineScorer, MatchScorer


class BlackBoxUQ(UncertaintyQuantifier):
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        scorers: Optional[List[str]] = None,
        device: Any = None,
        use_best: bool = True,
        nli_model_name: str = "microsoft/deberta-large-mnli",
        postprocessor: Any = None,
        system_prompt: str = "You are a helpful assistant.",
        max_calls_per_min: Optional[int] = None,
        sampling_temperature: float = 1.0,
        use_n_param: bool = False,
        max_length: int = 2000,
        verbose: bool = False,
    ) -> None:
        """
        Class for black box uncertainty quantification. Leverages multiple responses to the same prompt to evaluate
        consistency as an indicator of hallucination likelihood.

        Parameters
        ----------
        llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        scorers : subset of {
            'semantic_negentropy', 'noncontradiction', 'exact_match', 'bert_score', 'bleurt', 'cosine_sim'
        }, default=None
            Specifies which black box (consistency) scorers to include. If None, defaults to
            ["semantic_negentropy", "noncontradiction", "exact_match", "cosine_sim"].

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. Only applies to 'semantic_negentropy', 'noncontradiction'
            scorers. Pass a torch.device to leverage GPU.

        use_best : bool, default=True
            Specifies whether to swap the original response for the uncertainty-minimized response
            based on semantic entropy clusters. Only used if `scorers` includes 'semantic_negentropy' or 'noncontradiction'.

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        postprocessor : callable, default=None
            A user-defined function that takes a string input and returns a string. Used for postprocessing
            outputs.

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

        verbose : bool, default=False
            Specifies whether to print the index of response currently being scored.
        """
        super().__init__(llm=llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param, postprocessor=postprocessor)
        self.prompts = None
        self.max_length = max_length
        self.verbose = verbose
        self.use_best = use_best
        self.sampling_temperature = sampling_temperature
        self.nli_model_name = nli_model_name
        self._validate_scorers(scorers)
        self.use_nli = ("semantic_negentropy" in self.scorers) or ("noncontradiction" in self.scorers)
        if self.use_nli:
            self._setup_nli(nli_model_name)

    async def generate_and_score(self, prompts: List[str], num_responses: int = 5) -> UQResult:
        """
        Generate LLM responses, sampled LLM (candidate) responses, and compute confidence scores with specified scorers for the provided prompts.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        num_responses : int, default=5
            The number of sampled responses used to compute consistency.

        Returns
        -------
        UQResult
            UQResult containing data (prompts, responses, and scores) and metadata
        """
        self.prompts = prompts
        self.num_responses = num_responses

        responses = await self.generate_original_responses(prompts)
        sampled_responses = await self.generate_candidate_responses(prompts)
        return self.score(responses=responses, sampled_responses=sampled_responses)

    def score(self, responses: List[str], sampled_responses: List[List[str]]) -> UQResult:
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

        Returns
        -------
        UQResult
            UQResult containing data (prompts, responses, and scores) and metadata
        """
        print("Computing confidence scores...")
        self.responses = responses
        self.sampled_responses = sampled_responses
        self.num_responses = len(sampled_responses[0])

        self.scores_dict = {k: [] for k in self.scorer_objects}
        if self.use_nli:
            compute_entropy = "semantic_negentropy" in self.scorers
            nli_scores = self.nli_scorer.evaluate(responses=self.responses, sampled_responses=self.sampled_responses, use_best=self.use_best, compute_entropy=compute_entropy)
            if self.use_best:
                self.original_responses = self.responses.copy()
                self.responses = nli_scores["responses"]
                self.sampled_responses = nli_scores["sampled_responses"]

            for key in ["semantic_negentropy", "noncontradiction"]:
                if key in self.scorers:
                    if key == "semantic_negentropy":
                        nli_scores[key] = [1 - s for s in self.nli_scorer._normalize_entropy(nli_scores[key])]  # Convert to confidence score
                    self.scores_dict[key] = nli_scores[key]

        # similarity scorers that follow the same pattern
        for scorer_key in ["exact_match", "bert_score", "bleurt", "cosine_sim"]:
            if scorer_key in self.scorer_objects:
                self.scores_dict[scorer_key] = self.scorer_objects[scorer_key].evaluate(responses=self.responses, sampled_responses=self.sampled_responses)

        return self._construct_result()

    def _construct_result(self) -> Any:
        """Constructs UQResult object"""
        data = {"responses": self.responses, "sampled_responses": self.sampled_responses}
        if self.prompts:
            data["prompts"] = self.prompts
        data.update(self.scores_dict)
        result = {"data": data, "metadata": {"temperature": None if not self.llm else self.llm.temperature, "sampling_temperature": None if not self.sampling_temperature else self.sampling_temperature, "num_responses": self.num_responses, "scorers": self.scorers}}
        return UQResult(result)

    def _validate_scorers(self, scorers: List[Any]) -> None:
        "Validate scorers and construct applicable scorer attributes"
        self.scorer_objects = {}
        if scorers is None:
            scorers = self.default_black_box_names
        for scorer in scorers:
            if scorer == "exact_match":
                self.scorer_objects["exact_match"] = MatchScorer()
            elif scorer == "bert_score":
                self.scorer_objects["bert_score"] = BertScorer()
            elif scorer == "bleurt":
                from uqlm.black_box import BLEURTScorer

                self.scorer_objects["bleurt"] = BLEURTScorer()
            elif scorer == "cosine_sim":
                self.scorer_objects["cosine_sim"] = CosineScorer()
            elif scorer in ["semantic_negentropy", "noncontradiction"]:
                continue
            else:
                raise ValueError(
                    """
                    scorers must be one of ['semantic_negentropy', 'noncontradiction', 'exact_match', 'bert_score', 'bleurt', 'cosine_sim']
                    """
                )
        self.scorers = scorers
