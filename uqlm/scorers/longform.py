import warnings
from typing import Any, List, Optional
import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier
from uqlm.utils.results import UQResult
from uqlm.longform.black_box import LUQScorer
from uqlm.longform.decomposition import ResponseDecomposer

SENTENCE_BLACKBOX_SCORERS = ["response_sent_entail", "response_sent_noncontradict", "response_sent_contrast_entail"]
CLAIM_BLACKBOX_SCORERS = ["response_claim_entail", "response_claim_noncontradict", "response_claim_contrast_entail"]
MATCHED_CLAIM_BLACKBOX_SCORERS = ["matched_claim_entail", "matched_claim_noncontradict", "matched_claim_contrast_entail"]
NLI_SCORERS = SENTENCE_BLACKBOX_SCORERS + CLAIM_BLACKBOX_SCORERS + MATCHED_CLAIM_BLACKBOX_SCORERS
DATACLASS_NAMES = ["claim_entail_scores", "claim_noncontradict_scores", "claim_constrast_entail_scores"]
DATACLASS_TO_SCORER_MAP = {
    scorer: dataclass_name for scorer, dataclass_name in zip(NLI_SCORERS, DATACLASS_NAMES * 3)
}

class LongFormUQ(UncertaintyQuantifier):
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        scorers: Optional[List[str]] = None,
        claim_decomposition_llm: Optional[BaseChatModel] = None,
        device: Any = None,
        nli_model_name: str = "microsoft/deberta-large-mnli",
        system_prompt: str = "You are a helpful assistant.",
        max_calls_per_min: Optional[int] = None,
        sampling_temperature: float = 1.0,
        use_n_param: bool = False,
        max_length: int = 2000,
        postprocessor: Optional[Any] = None,
        return_responses: str = "all",
    ) -> None:
        """
        Class for longform uncertainty quantification. Implements claimwise analogs of other UQ-based
        scorers for improved performance with longform tasks.

        Parameters
        ----------
        llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        scorers : subset of {"luq", "luq_atomic"}, default=None
            Specifies which black box (consistency) scorers to include. If None, defaults to
            ["semantic_negentropy", "noncontradiction", "exact_match", "cosine_sim"].

        claim_decomposition_llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel` to be used for decomposing responses into individual claims
            or 'factoids'. If a scorer that requires claim decomposition (e.g., "luq_atomic") is specified
            and claim_decomposition_llm is None, the provided `llm` will be used for claim decomposition.

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. Applies to 'luq', 'luq_atomic'
            scorers. Pass a torch.device to leverage GPU.

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        postprocessor : callable, default=None
            A user-defined function that takes a string input and returns a string. Used for postprocessing
            outputs before black-box comparisons.

        return_responses : str, default="all"
            If a postprocessor is used, specifies whether to return only postprocessed responses, only raw responses,
            or both. Specified with 'postprocessed', 'raw', or 'all', respectively.

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

        super().__init__(llm=llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param, postprocessor=postprocessor)
        self.max_length = max_length
        self.sampling_temperature = sampling_temperature
        self.nli_model_name = nli_model_name
        self.claim_decomposition_llm = claim_decomposition_llm
        self.default_long_form_scorers = SENTENCE_BLACKBOX_SCORERS
        self.decomposer = ResponseDecomposer(claim_decomposition_llm=claim_decomposition_llm if claim_decomposition_llm else llm)
        self.prompts = None
        self.responses = None
        self.claim_sets = None
        self.sentence_sets = None
        self.sampled_responses = None
        self.num_responses = None
        self.luq_scorer = None
        self._validate_scorers(scorers)
        self.scores_dict = {}
        self.bb_claim_scores_dict = {}

    async def generate_and_score(self, prompts: List[str], num_responses: int = 5, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Generate LLM responses, sampled LLM (candidate) responses, and compute confidence scores with specified scorers for the provided prompts.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

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
        self._display_generation_header(show_progress_bars)

        responses = await self.generate_original_responses(prompts=prompts, progress_bar=self.progress_bar)
        sampled_responses = await self.generate_candidate_responses(prompts=prompts, progress_bar=self.progress_bar, num_responses=self.num_responses)
        result = await self.score(responses=responses, sampled_responses=sampled_responses, show_progress_bars=show_progress_bars)
        return result

    async def score(self, responses: List[str], sampled_responses: List[List[str]], show_progress_bars: Optional[bool] = True) -> UQResult:
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
        self._construct_progress_bar(show_progress_bars)
        
        await self._decompose_responses(show_progress_bars)
        self._display_scoring_header(show_progress_bars)
        
        self.scores_dict = {k: [] for k in self.scorers}
        if self.sentence_level_bb_scorers:
            self.bb_sentence_scores_dict = self.luq_scorer.evaluate(claim_sets=self.sentence_sets, sampled_responses=self.sampled_responses, progress_bar=self.progress_bar).to_dict()
            for scorer in self.sentence_level_bb_scorers:
                self.scores_dict[scorer] = [np.mean(claim_scores) for claim_scores in self.bb_sentence_scores_dict[DATACLASS_TO_SCORER_MAP[scorer]]]
        if self.claim_level_bb_scorers:
            claim_level_scores_result = self.luq_scorer.evaluate(claim_sets=self.claim_sets, sampled_responses=self.sampled_responses, progress_bar=self.progress_bar).to_dict()
            self.bb_claim_scores_dict.update(claim_level_scores_result)
            for scorer in self.claim_level_bb_scorers:
                self.scores_dict[scorer] = [np.mean(claim_scores) for claim_scores in self.bb_claim_scores_dict[DATACLASS_TO_SCORER_MAP[scorer]]]
        if self.matched_claim_bb_scorers:
            matched_claim_scores_result = self.luq_scorer.evaluate(claim_sets=self.claim_sets, sampled_claims=self.sampled_claim_sets, progress_bar=self.progress_bar).to_dict()
            self.bb_claim_scores_dict.update(matched_claim_scores_result)
            for scorer in self.matched_claim_bb_scorers:
                self.scores_dict[scorer] = [np.mean(claim_scores) for claim_scores in self.bb_claim_scores_dict[DATACLASS_TO_SCORER_MAP[scorer]]]                
        result = self._construct_result()
        
        self._stop_progress_bar()
        self.progress_bar = None
        return result

    async def _decompose_responses(self, show_progress_bars) -> None:
        """Display header and decompose responses"""
        self._display_decomposition_header(show_progress_bars)
        if self.sentence_level_bb_scorers:
            self.sentence_sets = self.decomposer.decompose_sentences(responses=self.responses, progress_bar=self.progress_bar)
        if self.claim_level_bb_scorers or self.matched_claim_bb_scorers:
            self.claim_sets = await self.decomposer.decompose_claims(responses=self.responses, progress_bar=self.progress_bar)        
        if self.matched_claim_bb_scorers:
            self.sampled_claim_sets = await self.decomposer.decompose_candidate_claims(sampled_responses=self.sampled_responses, progress_bar=self.progress_bar)    
         
    def _construct_result(self) -> Any:
        """Constructs UQResult object"""
        data = {"responses": self.responses, "sampled_responses": self.sampled_responses}
        if self.prompts:
            data["prompts"] = self.prompts
        if self.claim_sets:
            data["claim_sets"] = self.claim_sets
        if self.sentence_sets:
            data["sentence_sets"] = self.sentence_sets
        data.update(self.scores_dict)
        result = {"data": data, "metadata": {"temperature": None if not self.llm else self.llm.temperature, "sampling_temperature": None if not self.sampling_temperature else self.sampling_temperature, "num_responses": self.num_responses, "scorers": self.scorers}}
        return UQResult(result)
    
    def _display_decomposition_header(self, show_progress_bars: bool) -> None:
        """Displays decomposition header"""
        if show_progress_bars:
            self.progress_bar.start()
            self.progress_bar.add_task("")
            self.progress_bar.add_task("✂️ Decomposition")    

    def _validate_scorers(self, scorers: Optional[List[str]]) -> None:
        """Validate scorers"""
        self.scorer_objects = {}
        if scorers is None:
            scorers = self.default_long_form_scorers
        self.sentence_level_bb_scorers = set(SENTENCE_BLACKBOX_SCORERS).intersection(set(scorers))
        self.claim_level_bb_scorers = set(CLAIM_BLACKBOX_SCORERS).intersection(set(scorers))  
        self.matched_claim_bb_scorers = set(MATCHED_CLAIM_BLACKBOX_SCORERS).intersection(set(scorers))
        if sum(1 for s in [self.claim_level_bb_scorers, self.sentence_level_bb_scorers, self.matched_claim_bb_scorers] if s) > 1:
            warnings.warn("Redundant metrics configuration: Using more than one of sentence-level, claim-level, and matched-claim black-box scorers increases computation time without providing additional information. Use only one of these scorer types for substantially faster runtime.", UserWarning)
        for scorer in scorers:
            if scorer in NLI_SCORERS:
                if not self.luq_scorer:
                    self.luq_scorer = LUQScorer(nli_model_name=self.nli_model_name, device=self.device, max_length=self.max_length)
                else:
                    continue
            else:
                raise ValueError(
                    f"""
                    Invalid scorer: {scorer}. Must be one of {NLI_SCORERS}
                    """
                )
            self.scorers = scorers
