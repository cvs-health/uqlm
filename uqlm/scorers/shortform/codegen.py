from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.code import CodeEquivalence, CodeBLEU, VerbalizedConfidence, FunctionalEntropy, CosineScorer
from uqlm.scorers.shortform.white_box import WhiteBoxUQ
from uqlm.utils.results import UQResult
from uqlm.scorers.shortform.baseclass.uncertainty import ShortFormUQ


class CodeGenUQ(ShortFormUQ):
    def __init__(
        self, 
        llm: Optional[BaseChatModel] = None,
        system_prompt: Optional[str] = None,
        max_calls_per_min: Optional[int] = None,
        scorers: Optional[List[str]] = None,
        sampling_temperature: float = 1.0,
        top_k_logprobs: int = 15,
        length_normalize: bool = True,
        device: Any = None,
        max_length: int = 2000,
        sentence_transformer: str = "jinaai/jina-embeddings-v2-base-code",
        nli_model_name: str = "microsoft/deberta-large-mnli",
        lang: str = "python"
    ):
        """
        Class for computing confidence scores for code generation use cases.

        Parameters
        ----------
        llm : BaseChatModel
            A langchain llm object to get passed to chain constructor. User is responsible for specifying
            temperature and other relevant parameters to the constructor of their `llm` object.

        system_prompt : str, default=None
            Optional argument for user to provide custom system prompt. If prompts are list of strings and system_prompt is None,
            defaults to "You are a helpful assistant."

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.

        scorers : List[str], default=None
            Specifies which scorers to include. Must be subset of ["sequence_probability", "min_probability", "mean_token_negentropy", "min_token_negentropy", "probability_margin", "p_true", "consistency_and_confidence", "monte_carlo_probability", "codebleu", "code_equivalence", "verbalized_confidence", "functional_entropy", "semantic_sets", "cosine_sim"]. If None, defaults to all scorers.

        sampling_temperature : float, default=1.0
            The 'temperature' parameter for llm model to generate sampled LLM responses. Must be greater than 0.

        top_k_logprobs : int, default=15
            Specifies the number of logprobs to return for each response.

        length_normalize : bool, default=True
            Specifies whether to length normalize the logprobs.

        device : Any, default=None
            Specifies the device that NLI model use for prediction. If None, detects and returns the best available PyTorch device.
            Prioritizes CUDA (NVIDIA GPU), then MPS (macOS), then CPU.

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError

        sentence_transformer : str, default="jinaai/jina-embeddings-v2-base-code"
            Specifies which huggingface sentence transformer to use when computing cosine similarity for consistency_and_confidence. See
            https://huggingface.co/jinaai?sort_models=likes#models
            for more information. The recommended sentence transformer is 'jinaai/jina-embeddings-v2-base-code'.

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        lang : str, default="python"
            Specifies the language of the code, this is used while computing CodeBleu and CodeEquivalence scores (if "codebleu" or "code_equivalence" is in scorers). 
            This might require user to install additional dependencies. Must be one of ["python", "java", "sql"].
        """
        super().__init__(llm=llm, max_calls_per_min=max_calls_per_min, system_prompt=system_prompt)
        self.scorers = scorers
        self.sampling_temperature = sampling_temperature
        self.top_k_logprobs = top_k_logprobs
        self.length_normalize = length_normalize
        self.max_length = max_length
        self.sentence_transformer = sentence_transformer
        self.nli_model_name = nli_model_name
        self.lang = lang
        self._validate_scorers()

    async def generate_and_score(self, prompts: List[str], num_responses: Optional[int] = 5) -> UQResult:
        self._construct_progress_bar(True)
        _ = await self.generate(prompts=prompts, num_responses=num_responses)
        results = await self.score(prompts=prompts, responses=self.responses, sampled_responses=self.sampled_responses, logprobs_results=self.logprobs, sampled_logprobs_results=self.multiple_logprobs)
        return results

    async def generate(self, prompts: List[str], num_responses: Optional[int] = 5) -> UQResult:
        self.llm.logprobs = True
        self.responses = await self.generate_original_responses(prompts, top_k_logprobs=self.top_k_logprobs, progress_bar=self.progress_bar)
        self.sampled_responses = await self.generate_candidate_responses(prompts=prompts, num_responses=num_responses, progress_bar=self.progress_bar)
        
        data = {"prompts": prompts, "responses": self.responses, "sampled_responses": self.sampled_responses, "logprobs_results": self.logprobs, "sampled_logprobs_results": self.multiple_logprobs}
        return UQResult(result={"data": data})

    async def score(self, prompts: List[str], responses: List[str], sampled_responses: List[List[str]], logprobs_results: List[List[float]], sampled_logprobs_results: List[List[float]]) -> UQResult:
        data = {"prompts": prompts, "responses": responses, "logprob": logprobs_results, "sampled_responses": sampled_responses, "sampled_logprob": sampled_logprobs_results}
        data = {key: val for key, val in data.items() if val}
        
        # Compute Cosine Similarity
        if "cosine_sim" in self.scorers:
            data['cosine_sim'] = self.cos.evaluate(responses=responses, sampled_responses=sampled_responses)
            data["cosine_sim_pair_score"] = self.cos.pair_scores
            # if 'consistency_and_confidence' in self.scorers and 'sequence_probability' in self.scorers:
            #     data['cosine_sim'] = [data['consistency_and_confidence'][i]/data['sequence_probability'][i] for i in range(len(prompts))]

        # Compute White-box UQ scores
        if len(self.wbuq_scorers)>0:
            self.wbuq.progress_bar = self.progress_bar
            self.wb_results = await self.wbuq.score(prompts=prompts, responses=responses, sampled_responses=sampled_responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results)
            for key in self.wb_results.data:
                if key in self.scorers:
                    data[key] = self.wb_results.data[key]
        
        if "consistency_and_confidence" in self.scorers and "consistency_and_confidence" not in self.wbuq_scorers:
            data['consistency_and_confidence'] = [data['cosine_sim'][i]*data['sequence_probability'][i] for i in range(len(prompts))]
        
        # TODO: Remove this code block
        # # Compute Code Equivalence scores
        # if "code_equivalence" in self.scorers:
        #     data["code_equivalence"] = await self.ce.score(responses=responses, sampled_responses=sampled_responses)
        #     data["code_equivalence_cache"] = self.ce.equivalence_cache

        # Compute Code BLEU confidence scores
        if "codebleu" in self.scorers:
            data["codebleu"] = self.cb.score(responses=responses, sampled_responses=sampled_responses)
            data["code_bleu_pair_score"] = self.cb.pair_scores
        # Compute Verbalized Confidence scores
        if "verbalized_confidence" in self.scorers:
            data["verbalized_confidence"] = await self.vc.judge_responses(prompts=prompts, responses=responses, progress_bar=self.progress_bar)
        
        # TODO: Compute other scores here if functional entropy is not in scorers
        # Compute Functional Entropy scores
        if "functional_entropy" in self.scorers:
            # eq = self.ce.equivalence_cache if "code_equivalence" in self.scorers else None
            eq = None
            fe_results = await self.fe.score(responses=responses, sampled_responses=sampled_responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results, equivalence_indicators=eq)
            data["semantic_entropy"] = fe_results.data["discrete_confidence_scores"]
            data["tokenprob_semantic_entropy"] = fe_results.data["tokenprob_confidence_scores"]
            if "semantic_sets" in self.scorers:
                data["num_semantic_sets"] = fe_results.data["num_semantic_sets"]
            data["functional_entropy_equivalence_indicators"] = self.fe.equivalence_indicators

            data["semantic_negentropy"] = fe_results.data['discrete_confidence_scores']
            data["semantic_negentropy_whitebox"] = fe_results.data['tokenprob_confidence_scores']
            if "semantic_sets" in self.scorers:
                data["semantic_sets_confidence"] = fe_results.data['semantic_sets_confidence']
                data["cluster_indices"] = fe_results.data['cluster_indices']
            if "code_equivalence" in self.scorers:
                data["equivalence_rate"] = fe_results.data['equivalence_rate']
                data["original_equivalence_scores"] = fe_results.data['original_equivalence_scores'] 
        return UQResult(result={"data": data})

    def _validate_scorers(self):
        default_white_box_scorers = ["sequence_probability", "min_probability", "mean_token_negentropy", "min_token_negentropy", "probability_margin", "p_true", "consistency_and_confidence", "monte_carlo_probability"]
        default_judge_scorers = ["code_equivalence", "codebleu", "verbalized_confidence"]
        deafult_black_box_scorers = ["functional_entropy", "semantic_sets", "cosine_sim"]
        if not self.scorers:
            self.scorers = default_white_box_scorers + default_judge_scorers + deafult_black_box_scorers
        self.wbuq_scorers = []
        for scorer in self.scorers:
            if scorer in default_white_box_scorers:
                if scorer == "consistency_and_confidence" and "cosine_sim" in self.scorers and "sequence_probability" in self.scorers:
                    continue
                self.wbuq_scorers.append(scorer)

        if len(self.wbuq_scorers)>0:
            self.wbuq = WhiteBoxUQ(llm=self.llm, scorers=self.wbuq_scorers, system_prompt=self.system_prompt, max_calls_per_min=self.max_calls_per_min,
                                  sampling_temperature=self.sampling_temperature, top_k_logprobs=self.top_k_logprobs, 
                                  length_normalize=self.length_normalize, prompts_in_nli=False, 
                                  sentence_transformer=self.sentence_transformer, nli_model_name=self.nli_model_name)
        if "code_equivalence" in self.scorers:
            self.ce = CodeEquivalence(llm=self.llm, system_prompt=self.system_prompt, language=self.lang)
        if "codebleu" in self.scorers:
            self.cb = CodeBLEU(lang=self.lang)
        if "verbalized_confidence" in self.scorers:
            self.vc = VerbalizedConfidence(llm=self.llm, max_calls_per_min=self.max_calls_per_min)
        if "functional_entropy" in self.scorers:
            self.fe = FunctionalEntropy(equivalence_llm=self.llm, llm=self.llm, system_prompt=self.system_prompt)
        if "cosine_sim" in self.scorers:
            self.cos = CosineScorer(transformer=self.sentence_transformer)