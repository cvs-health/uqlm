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
from langchain_core.messages import BaseMessage
from typing import Any, List, Optional, Union

from uqlm.utils.results import UQResult
from uqlm.black_box import BertScorer, CosineScorer, MatchScorer, ConsistencyScorer
from uqlm.scorers.shortform.entropy import SemanticEntropy
from uqlm.scorers.shortform.baseclass.uncertainty import ShortFormUQ


class BlackBoxUQ(ShortFormUQ):
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        scorers: Optional[List[str]] = None,
        device: Any = None,
        use_best: bool = True,
        nli_model_name: str = "microsoft/deberta-large-mnli",
        sentence_transformer: str = "all-MiniLM-L6-v2",
        postprocessor: Any = None,
        system_prompt: Optional[str] = None,
        max_calls_per_min: Optional[int] = None,
        sampling_temperature: float = 1.0,
        return_responses: str = "all",
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

        scorers : List[str], default=None
            Specifies which black box (consistency) scorers to include. If None, defaults to
            ["semantic_negentropy", "noncontradiction", "exact_match", "cosine_sim", "entailment", "semantic_sets_confidence"]. The bleurt
            scorer is deprecated as of v0.2.0.

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. Only applies to 'semantic_negentropy', 'noncontradiction', 'entailment', 'semantic_sets_confidence'
            scorers. If None, detects and returns the best available PyTorch device. Prioritizes CUDA (NVIDIA GPU), then MPS (macOS), then CPU.

        use_best : bool, default=True
            Specifies whether to swap the original response for the uncertainty-minimized response
            based on semantic entropy clusters. Only used if `scorers` includes 'semantic_negentropy', 'noncontradiction', 'entailment', or 'semantic_sets_confidence'.

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        sentence_transformer : str, default="all-MiniLM-L6-v2"
            Specifies which huggingface sentence transformer to use when computing cosine similarity. See
            https://huggingface.co/sentence-transformers?sort_models=likes#models
            for more information. The recommended sentence transformer is 'all-MiniLM-L6-v2'.

        postprocessor : callable, default=None
            A user-defined function that takes a string input and returns a string. Used for postprocessing
            outputs before black-box comparisons.

        return_responses : str, default="all"
            If a postprocessor is used, specifies whether to return only postprocessed responses, only raw responses,
            or both. Specified with 'postprocessed', 'raw', or 'all', respectively.

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
        self.sentence_transformer = sentence_transformer
        self.return_responses = return_responses
        self.scorer_names = scorers
        self.generation_type = "default"
        self._validate_scorers(scorers)

    async def generate_and_score(self, prompts: List[Union[str, List[BaseMessage]]], num_responses: int = 5, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Generate LLM responses, sampled LLM (candidate) responses, and compute confidence scores with specified scorers for the provided prompts.

        Parameters
        ----------
        prompts : List[Union[str, List[BaseMessage]]]
            List of prompts from which LLM responses will be generated. Prompts in list may be strings or lists of BaseMessage. If providing
            input type List[List[BaseMessage]], refer to https://python.langchain.com/docs/concepts/messages/#langchain-messages for support.

        num_responses : int, default=5
            The number of sampled responses used to compute consistency.

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
        self._display_generation_header(show_progress_bars, generation_type=self.generation_type)

        responses = await self.generate_original_responses(prompts=prompts, progress_bar=self.progress_bar)
        sampled_responses = await self.generate_candidate_responses(prompts=prompts, num_responses=self.num_responses, progress_bar=self.progress_bar)
        result = self.score(responses=responses, sampled_responses=sampled_responses, show_progress_bars=show_progress_bars)
        return result

    def score(self, responses: List[str], sampled_responses: List[List[str]], show_progress_bars: Optional[bool] = True, _display_header: bool = True) -> UQResult:
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
        self.scores_dict = dict()

        self._construct_progress_bar(show_progress_bars)
        self._display_scoring_header(show_progress_bars and _display_header)

        available_nli_scores = dict()
        if self.entropy_scorer_names:
            self.scorer_objects["semantic_negentropy"].progress_bar = self.progress_bar
            se_tmp = self.scorer_objects["semantic_negentropy"].score(responses=self.responses, sampled_responses=self.sampled_responses, _display_header=False)
            if "semantic_negentropy" in self.scorer_names:
                self.scores_dict["semantic_negentropy"] = [1 - s for s in self.scorer_objects["semantic_negentropy"]._normalize_entropy(se_tmp.data["discrete_entropy_values"])]  # Convert to confidence score
            if "semantic_sets_confidence" in self.scorer_names:
                self.scores_dict["semantic_sets_confidence"] = [1 - (s - 1) / (self.num_responses - 1) for s in se_tmp.data["num_semantic_sets"]]  # Convert to confidence score
            available_nli_scores = self.scorer_objects["semantic_negentropy"].clusterer.nli_scores
            if self.use_best:
                self._update_best(se_tmp.data["responses"], include_logprobs=False)

        self._start_progress_bar()  # restart progress bar as entropy scorer stops it
        if self.consistency_scorer_names:
            self.scorer_objects["consistency"].use_best = False
            consistency_tmp = self.scorer_objects["consistency"].evaluate(responses=self.responses, sampled_responses=self.sampled_responses, available_nli_scores=available_nli_scores, progress_bar=self.progress_bar)
            for scorer in self.consistency_scorer_names:
                self.scores_dict[scorer] = consistency_tmp[scorer]

        for scorer in self.similarity_scorer_names:
            self.scores_dict[scorer] = self.scorer_objects[scorer].evaluate(responses=self.responses, sampled_responses=self.sampled_responses, progress_bar=self.progress_bar)

        result = self._construct_result()

        self._stop_progress_bar()
        self.progress_bar = None  # if re-run ensure the same progress object is not used
        return result

    def _construct_result(self) -> Any:
        """Constructs UQResult object"""
        data_to_return = self._construct_black_box_return_data()
        data_to_return.update(self.scores_dict)
        result = {"data": data_to_return, "metadata": {"temperature": None if not self.llm else self.llm.temperature, "sampling_temperature": None if not self.sampling_temperature else self.sampling_temperature, "num_responses": self.num_responses, "scorers": self.scorers}}
        return UQResult(result)

    def _validate_scorers(self, scorers: List[Any]) -> None:
        "Validate scorers and construct applicable scorer attributes"
        self.scorer_objects = {}
        if scorers is None:
            scorers = self.default_black_box_names
        self.consistency_scorer_names, self.entropy_scorer_names, self.similarity_scorer_names = [], [], []
        for scorer in scorers:
            if scorer == "exact_match":
                self.scorer_objects["exact_match"] = MatchScorer()
                self.similarity_scorer_names.append(scorer)
            elif scorer == "bert_score":
                self.scorer_objects["bert_score"] = BertScorer(device=self.device)
                self.similarity_scorer_names.append(scorer)
            elif scorer == "cosine_sim":
                self.scorer_objects["cosine_sim"] = CosineScorer(transformer=self.sentence_transformer)
                self.similarity_scorer_names.append(scorer)
            elif scorer in ["noncontradiction", "entailment"]:
                self.consistency_scorer_names.append(scorer)
            elif scorer in ["semantic_negentropy", "semantic_sets_confidence"]:
                self.entropy_scorer_names.append(scorer)
            else:
                if scorer == "bleurt":
                    print("bleurt is deprecated as of v0.2.0")
                raise ValueError(
                    """
                    scorers must be one of ['semantic_negentropy', 'noncontradiction', 'exact_match', 'bert_score', 'cosine_sim', 'semantic_sets_confidence', 'entailment']
                    """
                )
        if self.consistency_scorer_names:
            self.scorer_objects["consistency"] = ConsistencyScorer(nli_model_name=self.nli_model_name, max_length=self.max_length, use_best=self.use_best, scorers=self.consistency_scorer_names)
        if self.entropy_scorer_names:
            self.scorer_objects["semantic_negentropy"] = SemanticEntropy(llm=self.llm, nli_model_name=self.nli_model_name, max_length=self.max_length, use_best=self.use_best)
        self.scorers = scorers
