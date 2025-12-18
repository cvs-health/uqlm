from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
from rich.progress import Progress
from uqlm.black_box.baseclass.similarity_scorer import SimilarityScorer
from uqlm.nli.nli import NLI
from uqlm.nli.cluster import SemanticClusterer


class ConsistencyScorer(SimilarityScorer):
    def __init__(self, nli_model_name: str = "microsoft/deberta-large-mnli", max_length: int = 2000, use_best: bool = False, scorers: List[str] = ["noncontradiction", "entailment"]):
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
        self.nli = NLI(nli_model_name=nli_model_name, max_length=max_length)
        self.scorers = scorers

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
        observed_consistency_data = {"noncontradiction": [], "entailment": [], "discrete_semantic_entropy": [], "tokenprob_semantic_entropy": [], "responses": responses, "sampled_responses": sampled_responses}

        def _process_i(i, response):
            oc_result_i = self._observed_consistency_i(original=response, candidates=sampled_responses[i])
            for scorer in self.scorers:
                observed_consistency_data[scorer].append(oc_result_i[scorer])
            responses[i] = oc_result_i["response"]  # Replace with optimized response if use_best
            sampled_responses[i] = oc_result_i["candidates"]  # Replace with updated candidates if use_best

        if progress_bar:
            progress_task = progress_bar.add_task("  - Scoring responses with entailment/contradiction...", total=len(responses))
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

            self.clusterer = SemanticClusterer(nli=self.nli)
            _, response_probabilities = self.clusterer.compute_response_probabilities(logprobs_results=None, num_responses=len(all_responses))
            best_response, _, _, _ = self.clusterer.evaluate(responses=all_responses, response_probabilities=response_probabilities)

            candidates = all_responses.remove(best_response)
            self.available_nli_scores = self.clusterer.nli_scores

        nli_scores = {}
        for s_ in self.scorers:
            nli_scores[s_] = []
            for candidate in candidates:
                if s_ in self.available_nli_scores:
                    if (candidate, best_response) in self.available_nli_scores[s_]:
                        nli_scores[s_].append(self.available_nli_scores[s_][(candidate, best_response)])
                        continue
                nli_scores[s_].append(self.nli.get_nli_results(response1=best_response, response2=candidate)[s_ + "_score"])

        result = {n: np.mean(nli_scores[n]) for n in self.scorers}
        result.update({"candidates": candidates, "response": best_response})
        return result
