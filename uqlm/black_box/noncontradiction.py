from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
from rich.progress import Progress
from uqlm.black_box.baseclass.similarity_scorer import SimilarityScorer
from uqlm.nli.nli import NLIScorer
from uqlm.utils.cluster import Cluster


class NonContradictionScorer(SimilarityScorer):
    def __init__(self, nli_model_name: str = "microsoft/deberta-large-mnli", max_length: int = 2000, use_best: bool = False):
        """
        Initialize the NonContradictionScorer.

        Parameters
        ----------
        use_best : bool, default=False
            Specifies whether to swap the original response for the uncertainty-minimized response
            based on semantic entropy clusters.
        """
        super().__init__()
        self.nli_model_name = nli_model_name
        self.max_length = max_length
        self.use_best = use_best
        self.nli_scorer = NLIScorer(nli_model_name=nli_model_name, max_length=max_length)

    def evaluate(self, responses: List[str], sampled_responses: List[List[str]], available_nli_scores: Dict[Tuple[str, str], float] = dict(), progress_bar: Optional[Progress] = None) -> Dict[str, Any]:
        """
        Evaluate confidence scores on LLM responses.

        Parameters
        ----------
        responses : list of strings
            Original LLM response

        sampled_responses : list of list of strings
            Sampled candidate responses to be compared to the original response

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses

        Returns
        -------
        Dict
            Dictionary containing mean NLI and (optionally) semantic entropy scores.
            The dictionary will also contain original and multiple responses, updated if `use_best` is True
        """
        self.available_nli_scores = available_nli_scores
        self.num_responses = len(sampled_responses[0])
        observed_consistency_data = {"noncontradiction": [], "discrete_semantic_entropy": [], "tokenprob_semantic_entropy": [], "responses": responses, "sampled_responses": sampled_responses}

        def _process_i(i, response):
            oc_result_i = self._observed_consistency_i(original=response, candidates=sampled_responses[i])
            observed_consistency_data["noncontradiction"].append(oc_result_i["nli_score_i"])
            responses[i] = oc_result_i["response"]  # Replace with optimized response if use_best
            sampled_responses[i] = oc_result_i["candidates"]  # Replace with updated candidates if use_best

        if progress_bar:
            progress_task = progress_bar.add_task("  - Scoring responses with NLI...", total=len(responses))
        for i, response in enumerate(responses):
            _process_i(i, response)
            if progress_bar:
                progress_bar.update(progress_task, advance=1)
        time.sleep(0.1)

        if self.use_best:
            observed_consistency_data["responses"] = responses
            observed_consistency_data["sampled_responses"] = sampled_responses
        return observed_consistency_data

    def _observed_consistency_i(self, original: str, candidates: List[str]) -> Dict[str, Any]:
        """
        Compute observed consistency score on the provided original response and multiple candidates.
        """
        best_response = original
        if self.use_best:
            all_responses = [original] + candidates

            self.cluster = Cluster(nli_scorer=self.nli_scorer)
            _, response_probabilities = self.cluster.compute_response_probabilities(logprobs_results=None, num_responses=len(all_responses))
            best_response, _, _, _ = self.cluster.evaluate(responses=all_responses, response_probabilities=response_probabilities)

            candidates = all_responses.remove(best_response)
            self.available_nli_scores = self.cluster.nli_scores
            
        nli_scores = []
        for candidate in candidates:
            if (candidate, best_response) in self.available_nli_scores:
                nli_score = self.available_nli_scores[(candidate, best_response)]
            else:
                nli_score = self.nli_scorer.get_nli_results(response1=best_response, response2=candidate)["score"]
            nli_scores.append(nli_score)

        return {"nli_score_i": np.mean(nli_scores), "candidates": candidates, "response": best_response}
