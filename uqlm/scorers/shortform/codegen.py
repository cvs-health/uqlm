from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
import numpy as np
from uqlm.code import CodeBLEU, VerbalizedConfidence, FunctionalEntropy
from uqlm.black_box import CosineScorer
from uqlm.scorers.shortform.white_box import WhiteBoxUQ
from uqlm.utils.results import UQResult
from uqlm.scorers.shortform.baseclass.uncertainty import ShortFormUQ


DEFAULT_SCORERS = ["functional_equivalence_rate", "cosine_sim"]
WHITE_BOX_SCORERS = ["sequence_probability", "min_probability", "mean_token_negentropy", "min_token_negentropy", "probability_margin", "p_true", "consistency_and_confidence", "monte_carlo_probability"]
FUNCTIONAL_EQUIVALENCE_SCORERS = ["functional_negentropy", "functional_sets_confidence", "functional_equivalence_rate"]
OTHER_SCORERS = ["cosine_sim", "code_bleu", "verbalized_confidence"]
ALL_SCORERS = WHITE_BOX_SCORERS + OTHER_SCORERS + FUNCTIONAL_EQUIVALENCE_SCORERS


class CodeGenUQ(ShortFormUQ):
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        scorers: Optional[List[str]] = None,
        equivalence_llm: Optional[BaseChatModel] = None,
        system_prompt: Optional[str] = None,
        max_calls_per_min: Optional[int] = None,
        sampling_temperature: float = 1.0,
        top_k_logprobs: int = 15,
        length_normalize: bool = True,
        max_length: int = 2000,
        sentence_transformer: str = "jinaai/jina-embeddings-v2-base-code",
        language: str = "python",
        retries: int = 5,
    ):
        """
        Class for computing confidence scores for code generation use cases.

        Parameters
        ----------
        llm : BaseChatModel
            A langchain llm object to get passed to chain constructor. User is responsible for specifying
            temperature and other relevant parameters to the constructor of their `llm` object.

        scorers : List[str], default=None
            Specifies which scorers to include. Must be subset of ["sequence_probability", "min_probability", "mean_token_negentropy",
            "min_token_negentropy", "probability_margin", "p_true", "consistency_and_confidence", "monte_carlo_probability", "code_bleu",
            "functional_equivalence_rate", "verbalized_confidence", "functional_negentropy", "functional_sets_confidence", "cosine_sim"].
            If None, defaults to ["functional_equivalence_rate", "cosine_sim"].

        equivalence_llm : BaseChatModel, default=None
            A langchain llm object to get passed to chain constructor. This is used for CodeEquivalence and FunctionalEntropy scorers. User is responsible for specifying
            temperature and other relevant parameters to the constructor of their `equivalence_llm` object.

        system_prompt : str, default=None
            Optional argument for user to provide custom system prompt. If prompts are list of strings and system_prompt is None,
            defaults to "You are a helpful assistant."

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.

        sampling_temperature : float, default=1.0
            The 'temperature' parameter for llm model to generate sampled LLM responses. Must be greater than 0.

        top_k_logprobs : int, default=15
            Specifies the number of logprobs to return for each response.

        length_normalize : bool, default=True
            Specifies whether to length normalize the logprobs.

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError

        sentence_transformer : str, default="jinaai/jina-embeddings-v2-base-code"
            Specifies which huggingface sentence transformer to use when computing cosine similarity for consistency_and_confidence. See
            https://huggingface.co/jinaai?sort_models=likes#models
            for more information. The recommended sentence transformer is 'jinaai/jina-embeddings-v2-base-code'.

        language : str, default="python"
            Specifies the language of the code, this is used while computing CodeBleu and CodeEquivalence scores (if "codebleu" or "functional_equivalence_rate" is in scorers).
            This might require user to install additional dependencies. Must be one of ["python", "java", "sql"].

        retries : int, default=5
            Specifies the number of retries to make if the equivalence score is not found.
        """
        super().__init__(llm=llm, max_calls_per_min=max_calls_per_min, system_prompt=system_prompt)
        self.scorers = scorers
        self.sampling_temperature = sampling_temperature
        self.top_k_logprobs = top_k_logprobs
        self.length_normalize = length_normalize
        self.max_length = max_length
        self.sentence_transformer = sentence_transformer
        self.language = language
        self.equivalence_llm = equivalence_llm if equivalence_llm else llm
        self.retries = retries
        self.generation_type = "default"
        self._validate_scorers()

    async def generate_and_score(self, prompts: List[str], num_responses: Optional[int] = 5, show_progress_bars: Optional[bool] = True) -> UQResult:
        # self._construct_progress_bar(True)
        self._construct_progress_bar(show_progress_bars)
        self._display_generation_header(show_progress_bars, generation_type=self.generation_type)
        if self.wbuq_scorers:
            self.llm.logprobs = True
        self.responses = await self.generate_original_responses(prompts, top_k_logprobs=self.top_k_logprobs, progress_bar=self.progress_bar)
        self.sampled_responses = await self.generate_candidate_responses(prompts=prompts, num_responses=num_responses, progress_bar=self.progress_bar)
        results = await self.score(prompts=prompts, responses=self.responses, sampled_responses=self.sampled_responses, logprobs_results=self.logprobs, sampled_logprobs_results=self.multiple_logprobs, show_progress_bars=show_progress_bars)
        return results

    async def score(self, prompts: List[str], responses: List[str], sampled_responses: List[List[str]], logprobs_results: List[List[float]], sampled_logprobs_results: List[List[float]], show_progress_bars: Optional[bool] = True, _display_header: bool = True) -> UQResult:
        data = {"prompts": prompts, "responses": responses, "sampled_responses": sampled_responses}
        if logprobs_results[0] is not None:
            data["logprob"] = sampled_logprobs_results
        if sampled_logprobs_results[0][0] is not None:
            data["sampled_logprob"] = sampled_logprobs_results

        data = {key: val for key, val in data.items() if val}

        self._display_scoring_header(show_progress_bars and _display_header)

        # Compute Verbalized Confidence scores
        if "verbalized_confidence" in self.scorers:
            data["verbalized_confidence"] = await self.vc.judge_responses(prompts=prompts, responses=responses, progress_bar=self.progress_bar)

        # Compute Cosine Similarity
        if "cosine_sim" in self.scorers:
            data["cosine_sim"] = self.cos.evaluate(responses=responses, sampled_responses=sampled_responses, progress_bar=self.progress_bar)

        # Compute White-box UQ scores
        if len(self.wbuq_scorers) > 0:
            self.wbuq.progress_bar = self.progress_bar
            self.wb_results = await self.wbuq.score(prompts=prompts, responses=responses, sampled_responses=sampled_responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results, _display_header=False)
            for key in self.wb_results.data:
                if key in self.scorers:
                    data[key] = self.wb_results.data[key]

        if "consistency_and_confidence" in self.scorers and "consistency_and_confidence" not in self.wbuq_scorers:
            data["consistency_and_confidence"] = [data["cosine_sim"][i] * data["sequence_probability"][i] for i in range(len(prompts))]

        # Compute Code BLEU confidence scores
        if "codebleu" in self.scorers:
            data["codebleu"] = self.cb.evaluate(responses=responses, sampled_responses=sampled_responses, progress_bar=self.progress_bar)

        # Compute Functional Entropy scores
        if self.functional_equivalence_scorers == ["functional_equivalence_rate"]:
            fe_scores = await self.fe.clusterer.get_equivalence_scores(responses=responses, sampled_responses=sampled_responses, progress_bar=self.progress_bar)
            data["functional_equivalence_rate"] = [np.mean(s) for s in fe_scores]

        elif self.functional_equivalence_scorers:
            fe_results = await self.fe.evaluate(responses=responses, sampled_responses=sampled_responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results, progress_bar=self.progress_bar)
            for scorer in self.functional_equivalence_scorers:
                data[scorer] = fe_results[scorer]
            if "functional_negentropy_whitebox" in fe_results:
                data["functional_negentropy_whitebox"] = fe_results["functional_negentropy_whitebox"]

        self._stop_progress_bar()
        self.progress_bar = None  # if re-run ensure the same progress object is not used

        return UQResult(result={"data": data})

    def _validate_scorers(self):
        if not self.scorers:
            self.scorers = DEFAULT_SCORERS

        if not set(self.scorers).issubset(set(ALL_SCORERS)):
            raise ValueError(f"Invalid scorers: {list(set(self.scorers) - set(ALL_SCORERS))}")

        if "consistency_and_confidence" in self.scorers:
            self.scorers = list(set(self.scorers) & set(["cosine_sim", "sequence_probability"]))
        if "codebleu" in self.scorers:
            self.cb = CodeBLEU(language=self.language)
        if "verbalized_confidence" in self.scorers:
            self.vc = VerbalizedConfidence(llm=self.llm, max_calls_per_min=self.max_calls_per_min)
        if "cosine_sim" in self.scorers:
            self.cos = CosineScorer(transformer=self.sentence_transformer, max_length=self.max_length)

        self.wbuq_scorers = list(set(WHITE_BOX_SCORERS) & set(self.scorers))
        if len(self.wbuq_scorers) > 0:
            self.wbuq = WhiteBoxUQ(llm=self.llm, scorers=self.wbuq_scorers, system_prompt=self.system_prompt, max_calls_per_min=self.max_calls_per_min, sampling_temperature=self.sampling_temperature, top_k_logprobs=self.top_k_logprobs, length_normalize=self.length_normalize, prompts_in_nli=False, sentence_transformer=self.sentence_transformer)

        self.functional_equivalence_scorers = list(set(FUNCTIONAL_EQUIVALENCE_SCORERS) & set(self.scorers))
        if self.functional_equivalence_scorers:
            self.fe = FunctionalEntropy(equivalence_llm=self.equivalence_llm, system_prompt=self.system_prompt, length_normalize=self.length_normalize, language=self.language, retries=self.retries)
