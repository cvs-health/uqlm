from typing import Any, Dict, List, Optional, Tuple
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.scorers.long_form.baseclass.uncertainty import LongFormUQ
from uqlm.utils.results import UQResult
from uqlm.longform.graph import GraphScorer, ClaimMerger

GRAPH_SCORERS = ["degree_centrality", "betweenness_centrality", "closeness_centrality", "page_rank", "laplacian_centrality", "harmonic_centrality"]

class ClaimGraphUQ(LongFormUQ):
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        scorers: Optional[List[str]] = None,
        aggregation: str = "mean",
        claim_refinement: bool = False,
        claim_refinement_threshold: float = 1 / 3,
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
            Specifies which graph-based scorers to include. Must be subset of ["degree_centrality", "betweenness_centrality", 
            "closeness_centrality", "page_rank", "laplacian_centrality", "harmonic_centrality"]. If None, defaults to ["closeness_centrality"].

        granularity : str, default="claim"
            Specifies whether to decompose and score at claim or sentence level granularity. Must be either "claim" or "sentence"

        aggregation : str, default="mean"
            Specifies how to aggregate claim/sentence-level scores to response-level scores. Must be one of 'min' or 'mean'.

        claim_refinement : bool, default=False
            Specifies whether to refine responses with uncertainty-aware decoding. This approach removes claims with confidence
            scores below the claim_refinement_threshold and uses the claim_decomposition_llm to reconstruct the response from
            the retained claims. Only available for claim-level granularity. For more details, refer to
            Jiang et al., 2024: https://arxiv.org/abs/2410.20783

        claim_refinement_threshold : float, default=1/3
            Threshold for uncertainty-aware filtering. Claims with confidence scores below this threshold are dropped from the
            refined response. Only used if claim_refinement is True.

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
        self.scorers = ["closeness_centrality"] if not scorers else scorers
        super().__init__(llm=llm, aggregation=aggregation, scorers=self.scorers, claim_refinement=claim_refinement, claim_refinement_threshold=claim_refinement_threshold, claim_decomposition_llm=claim_decomposition_llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param)
        self.nli_model_name = nli_model_name
        self.max_length = max_length
        self.sampling_temperature = sampling_temperature
        self.graph_scorer = GraphScorer(nli_model_name=nli_model_name, max_length=max_length, device=device)
        self.claim_merger = ClaimMerger(claim_merging_llm=self.decomposer.claim_decomposition_llm)
        self.prompts = None
        self.responses = None
        self.claim_sets = None
        self.sampled_responses = None
        self.sampled_claim_sets = None
        self.num_responses = None
        self.uad_result = {}

    async def generate_and_score(self, prompts: List[str], num_responses: int = 5, claim_refinement_threshold: float = 1 / 3, show_progress_bars: Optional[bool] = True) -> UQResult:
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

    async def score(self, responses: List[str], sampled_responses: List[List[str]], claim_refinement_threshold: float = 1 / 3, show_progress_bars: Optional[bool] = True) -> UQResult:
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
        await self._decompose_candidate_responses(show_progress_bars)
        await self._merge_claims(show_progress_bars)
        
        self._display_scoring_header(show_progress_bars)
        
        all_responses = [[r] + sr for r, sr in zip(self.responses, self.sampled_responses)]

        original_claim_scores, master_claim_scores, graph_score_result = self._score_from_decomposed(original_claim_sets=self.claim_sets, master_claim_sets=self.master_claim_sets, response_sets=all_responses, progress_bar=self.progress_bar)
        
        if self.claim_refinement:
            self.claim_scores = master_claim_scores
            self.uad_result = await self.uncertainty_aware_decode(claim_sets=self.master_claim_sets, uad_claim_scores=self.claim_scores[self.uad_scorer], show_progress_bars=show_progress_bars)
        self._stop_progress_bar()
        self.progress_bar = None
                
        self.scores_dict["claims_data"] = self._unpack_claims_data(graph_score_result)
        if "removed" in self.uad_result:
            del self.uad_result["removed"]

        return self._construct_result()

    def _score_from_decomposed(self, original_claim_sets: List[List[str]], master_claim_sets: List[List[str]], response_sets: List[List[str]], progress_bar: Optional[Progress] = None) -> Tuple[Any, Any, Any]:
        """
        Compute confidence scores with specified scorers on provided LLM responses. Should only be used if responses and sampled responses
        are already generated. Otherwise, use `generate_and_score`.

        Parameters
        ----------
        claim_sets : list of list of strings
            List of original responses decomposed into lists of either claims or sentences

        master_claim_sets : list of list of strings
            Candidate responses to be compared to the decomposed original responses

        response_sets : list of list of strings
            Decomposed responses to be compared to the decomposed original responses

        Returns
        -------
        UQResult
            UQResult containing data (responses and scores) and metadata
        """
        graph_score_result = self.graph_scorer.evaluate(
            original_claim_sets=self.claim_sets, master_claim_sets=self.master_claim_sets, response_sets=response_sets, progress_bar=progress_bar
        )
        original_claim_scores, master_claim_scores = self._unpack_results(graph_score_result)
        return original_claim_scores, master_claim_scores, graph_score_result

    def _construct_result(self) -> Any:
        """Constructs UQResult object"""
        data = {"responses": self.responses, "sampled_responses": self.sampled_responses}
        if self.prompts:
            data["prompts"] = self.prompts
        data.update(self.scores_dict)
        data.update(self.uad_result)
        result = {"data": data, "metadata": {"aggregation": self.aggregation, "temperature": None if not self.llm else self.llm.temperature, "sampling_temperature": None if not self.sampling_temperature else self.sampling_temperature, "num_responses": self.num_responses, "claim_refinement_threshold": self.claim_refinement_threshold}}
        return UQResult(result)

    async def _merge_claims(self, show_progress_bars) -> None:
        self.master_claim_sets = await self.claim_merger.merge_claims(
            original_claim_sets=self.claim_sets, 
            sampled_claim_sets=self.sampled_claim_sets,
            progress_bar=self.progress_bar if show_progress_bars else None,
        )
        
    def _unpack_results(self, result: List[List[Any]]) -> Any:        
        original_claim_scores = {k: [] for k in self.scorers}
        master_claim_scores = {k: [] for k in self.scorers}
        self.scores_dict = {k: [] for k in self.scorers}
        for scorer in self.scorers:
            for i in range(len(result)):
                score_list_i = []
                master_score_list_i = []
                for j in range(len(result[i])):
                    master_score_list_i.append(result[i][j].scores[scorer])
                    if result[i][j].original_response:
                        score_list_i.append(result[i][j].scores[scorer])   
                original_claim_scores[scorer].append(score_list_i)
                master_claim_scores[scorer].append(master_score_list_i)
                
            response_scores = self._aggregate_scores(original_claim_scores[scorer])
            self.scores_dict[scorer] = response_scores
        return original_claim_scores, master_claim_scores 
        
        
    def _unpack_claims_data(self, result: List[List[Any]]) -> List[List[Dict[str, Any]]]:
        claims_data = []    
        for i in range(len(result)):
            claims_data_i = []
            for j in range(len(result[i])):
                claim_dict_ij = result[i][j].dict() 
                scores_dict_ij = claim_dict_ij.pop("scores")
                claim_dict_ij.update({k: s for k, s in scores_dict_ij.items() if k in self.scorers})
                claim_dict_ij["removed"] = False if not self.uad_result else self.uad_result["removed"][i][j]
                
                claims_data_i.append(claim_dict_ij)
            claims_data.append(claims_data_i)
        return claims_data
