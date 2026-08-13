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


from typing import List, Dict, Any, Optional
import numpy as np
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.scorers.shortform.entropy import SemanticEntropy
from uqlm.scorers.shortform.density import SemanticDensity
from uqlm.black_box.cosine import CosineScorer
from uqlm.nli.nli import NLI
from uqlm.white_box.baseclass.logprobs_scorer import LogprobsScorer


SAMPLED_LOGPROBS_SCORER_NAMES = ["semantic_negentropy", "semantic_density", "monte_carlo_probability", "consistency_and_confidence"]


class SampledLogprobsScorer(LogprobsScorer):
    def __init__(self, scorers: List[str] = SAMPLED_LOGPROBS_SCORER_NAMES, llm: BaseChatModel = None, nli_model_name: str = "microsoft/deberta-large-mnli", max_length: int = 2000, prompts_in_nli: bool = True, length_normalize: bool = True, device: Any = None, sentence_transformer: str = "sentence-transformers/all-MiniLM-L6-v2", nli_batch_size: int = 32) -> None:
        """
        Initialize the SampledLogprobsScorer.

        Parameters
        ----------
        scorers : List[str], default=SAMPLED_LOGPROBS_SCORER_NAMES
            Specifies which scorers to compute.
            Must be a subset of ["semantic_negentropy", "semantic_density", "monte_carlo_probability", "consistency_and_confidence"].

        llm : BaseChatModel, default=None
            Specifies the LLM to use. Must be a BaseChatModel.

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError

        prompts_in_nli : bool, default=True
            Specifies whether to use the prompts in the NLI inputs for semantic entropy and semantic density scorers.

        length_normalize : bool, default=True
            Specifies whether to length normalize the logprobs. This attribute affect the response probability computation for three scorers (semantic_negentropy, semantic_density, and monte_carlo_probability).

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. Only applies to 'semantic_negentropy', 'semantic_density' scorers. Pass a torch.device to leverage GPU.

        sentence_transformer : str (HuggingFace sentence transformer), default='all-MiniLM-L6-v2'
            Specifies which huggingface sentence transformer to use when computing cosine similarity. See
            https://huggingface.co/sentence-transformers?sort_models=likes#models
            for more information. The recommended sentence transformer is 'sentence-transformers/all-MiniLM-L6-v2'.

        nli_batch_size : int, default=32
            Number of premise-hypothesis pairs scored per forward pass by the NLI model. Only applies to
            'semantic_negentropy' and 'semantic_density' scorers.
        """
        super().__init__()
        self.scorers = scorers
        self.llm = llm
        self.nli_model_name = nli_model_name
        self.max_length = max_length
        self.prompts_in_nli = prompts_in_nli
        self.length_normalize = length_normalize
        self.semantic_negentropy_scorer = None
        self.device = device
        self.sentence_transformer = sentence_transformer
        self.nli_batch_size = nli_batch_size
        # Heavyweight members are constructed lazily on first use and reused across evaluate() calls
        self._nli = None
        self._cosine_scorer = None
        self._semantic_density_scorer = None

    def evaluate(self, responses: List[str], sampled_responses: List[List[str]], logprobs_results: List[List[Dict[str, Any]]], sampled_logprobs_results: Optional[List[List[List[Dict[str, Any]]]]] = None, prompts: List[str] = None, progress_bar: Optional[Progress] = None):
        scores_dict = {}
        if "monte_carlo_probability" in self.scorers:
            scores_dict["monte_carlo_probability"] = self.monte_carlo_probability(responses=responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results)
        if "consistency_and_confidence" in self.scorers:
            scores_dict["consistency_and_confidence"] = self.compute_consistency_confidence(responses=responses, sampled_responses=sampled_responses, logprobs_results=logprobs_results, progress_bar=progress_bar)
        if "semantic_negentropy" in self.scorers:
            scores_dict["semantic_negentropy"] = self.compute_semantic_negentropy(responses=responses, prompts=prompts, sampled_responses=sampled_responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results, progress_bar=progress_bar)
        if "semantic_density" in self.scorers:
            scores_dict["semantic_density"] = self.compute_semantic_density(responses=responses, sampled_responses=sampled_responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results, prompts=prompts, progress_bar=progress_bar)
        return {k: scores_dict[k] for k in self.scorers}

    def _get_nli(self) -> NLI:
        """Load the NLI model once and share it across scorers and evaluate() calls"""
        if self._nli is None:
            self._nli = NLI(nli_model_name=self.nli_model_name, device=self.device, max_length=self.max_length, batch_size=self.nli_batch_size)
        return self._nli

    def compute_consistency_confidence(self, responses: List[str], sampled_responses: List[List[str]], logprobs_results: List[List[Dict[str, Any]]], progress_bar: Optional[Progress] = None) -> List[float]:
        if self._cosine_scorer is None:
            self._cosine_scorer = CosineScorer(transformer=self.sentence_transformer)
        cosine_scores = self._cosine_scorer.evaluate(responses=responses, sampled_responses=sampled_responses, progress_bar=progress_bar)
        score_fn = self._norm_prob if self.length_normalize else self._seq_prob
        response_probs = self._compute_single_generation_scores(logprobs_results, score_fn)
        cocoa_scores = [cs * rp for cs, rp in zip(cosine_scores, response_probs)]
        return cocoa_scores

    def monte_carlo_probability(self, responses: List[str], logprobs_results: List[List[Dict[str, Any]]], sampled_logprobs_results: List[List[List[Dict[str, Any]]]]) -> List[float]:
        monte_carlo_scores = []
        score_fn = self._norm_prob if self.length_normalize else self._seq_prob
        for i in range(len(responses)):
            all_logprobs_response_i = [logprobs_results[i]] + sampled_logprobs_results[i]
            all_sampled_sequence_probs_response_i = self._compute_single_generation_scores(all_logprobs_response_i, score_fn)
            monte_carlo_sequence_prob_i = np.mean(all_sampled_sequence_probs_response_i)
            monte_carlo_scores.append(monte_carlo_sequence_prob_i)
        return monte_carlo_scores

    def compute_semantic_negentropy(self, responses: List[str], prompts: List[str], sampled_responses: List[List[str]], logprobs_results: List[List[Dict[str, Any]]], sampled_logprobs_results: List[List[List[Dict[str, Any]]]], progress_bar: Optional[Progress] = None) -> List[float]:
        if self.semantic_negentropy_scorer is None:
            self.semantic_negentropy_scorer = SemanticEntropy(llm=self.llm, nli_model_name=self.nli_model_name, max_length=self.max_length, use_best=False, prompts_in_nli=self.prompts_in_nli, length_normalize=self.length_normalize, device=self.device, nli=self._get_nli())
        self.semantic_negentropy_scorer.prompts_in_nli = self.prompts_in_nli
        self.semantic_negentropy_scorer.progress_bar = progress_bar
        show_progress_bars = True if progress_bar else False
        se_result = self.semantic_negentropy_scorer.score(responses=responses, prompts=prompts, sampled_responses=sampled_responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results, show_progress_bars=show_progress_bars, _display_header=False)
        return se_result.to_dict()["data"]["tokenprob_confidence_scores"]

    def compute_semantic_density(self, responses: List[str], sampled_responses: List[List[str]], logprobs_results: List[List[Dict[str, Any]]], sampled_logprobs_results: List[List[List[Dict[str, Any]]]], prompts: List[str] = None, progress_bar: Optional[Progress] = None) -> List[float]:
        if self._semantic_density_scorer is None:
            # Shares the NLI instance (and its pair-probability cache) with the semantic entropy scorer
            self._semantic_density_scorer = SemanticDensity(llm=self.llm, nli_model_name=self.nli_model_name, max_length=self.max_length, length_normalize=self.length_normalize, device=self.device, nli=self._get_nli())
        semantic_density_scorer = self._semantic_density_scorer
        if self.semantic_negentropy_scorer:
            show_progress_bars = False
        else:
            semantic_density_scorer.progress_bar = progress_bar
            show_progress_bars = True
        sd_result = semantic_density_scorer.score(prompts=prompts, responses=responses, sampled_responses=sampled_responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results, show_progress_bars=show_progress_bars, _display_header=False)
        return sd_result.to_dict()["data"]["semantic_density_values"]
