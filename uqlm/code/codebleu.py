import math
import importlib
from typing import List, Optional
from rich.progress import Progress


class CodeBLEU:
    def __init__(self, language: str = "python"):
        """
        Class for computing CodeBLEU scores.

        Parameters
        ----------
        language : str
            Specifies the language of the code. Must be one of python, java, sql.
        """
        self.language = language
        # Check if codebleu is installed
        codebleu_spec = importlib.util.find_spec("codebleu")
        if codebleu_spec is None:
            raise ImportError("UQLM: codebleu is not installed or could not be imported. Please install it using `pip install git+https://github.com/k4black/codebleu.git#egg=codebleu', 'tree-sitter>=0.25', 'tree-sitter-python>=0.25'`")
        from codebleu import calc_codebleu

        self.calc_codebleu = calc_codebleu

    def evaluate(self, responses: List[str], sampled_responses: List[List[str]], progress_bar: Optional[Progress] = None) -> List[float]:
        """
        Evaluate CodeBLEU scores on the provided responses and sampled responses.

        Parameters
        ----------
        responses : List[str]
            List of responses to evaluate.
        sampled_responses : List[List[str]]
            List of lists of sampled responses to evaluate.

        Returns
        -------
        List[float]
            List of CodeBLEU scores.
        """
        if len(responses) == 0 or len(sampled_responses) == 0:
            raise ValueError("Either responses or sampled responses is empty")
        n_prompts = len(responses)
        self.scores, self.pair_scores = [0] * n_prompts, [[]] * n_prompts

        # Progress bar
        progress_task = None
        if progress_bar:
            progress_task = progress_bar.add_task("  - Scoring responses with CodeBleu....", total=n_prompts)

        for i in range(n_prompts):
            self.scores[i], self.pair_scores[i] = self.codebleu_confidence(responses[i], sampled_responses[i])

            if progress_bar and progress_task is not None:
                progress_bar.update(progress_task, advance=1)

        return self.scores

    def codebleu_confidence(self, response: str, sampled_responses: List[str]) -> float:
        """
        Calculate CodeBLEU confidence for a list of code strings.

        Parameters
        ----------
        response : str
            The response to evaluate.
        sampled_responses : List[str]
            List of sampled responses to evaluate.

        Returns
        -------
        float
            The CodeBLEU confidence score.
        """
        if not sampled_responses:
            return float("nan")

        tmp_scores = [self.codebleu_pair(response, candidate) for candidate in sampled_responses]
        tmp_scores_no_nan = [score for score in tmp_scores if not math.isnan(score)]
        score = float("nan") if not tmp_scores_no_nan else sum(tmp_scores_no_nan) / len(tmp_scores_no_nan)
        return score, tmp_scores

    def codebleu_pair(self, response: str, candidate: str) -> float:
        """
        Calculate CodeBLEU score for a pair of code strings.

        Parameters
        ----------
        response : str
            The response to evaluate.
        candidate : str
            The candidate response to evaluate.

        Returns
        -------
        float
            The CodeBLEU score.
        """
        if not response or not candidate:
            return float("nan")

        try:
            res = self.calc_codebleu([str(response)], [candidate], lang=self.language, weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
            return float(res["codebleu"])
        except Exception as e:
            print(e)
            return float("nan")
