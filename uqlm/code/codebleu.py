import math
import importlib
from typing import List


class CodeBLEU:
    def __init__(self, lang: str = "python"):
        self.lang = lang
        # Check if codebleu is installed
        codebleu_spec = importlib.util.find_spec("codebleu")
        if codebleu_spec is None:
            raise ImportError("UQLM: codebleu is not installed or could not be imported. Please install it using `pip install git+https://github.com/k4black/codebleu.git#egg=codebleu', 'tree-sitter>=0.25', 'tree-sitter-python>=0.25'`")
        from codebleu import calc_codebleu
        self.calc_codebleu = calc_codebleu

    def score(
        self, responses: List[str], sampled_responses: List[List[str]]
    ) -> List[float]:
        if len(responses) == 0 or len(sampled_responses) == 0:
            raise ValueError("Either responses or sampled responses is empty")
        n_prompts = len(responses)
        self.scores, self.pair_scores = [0] * n_prompts, [[]] * n_prompts
        for i in range(n_prompts):
            self.scores[i] = self.codebleu_confidence(
                responses[i], sampled_responses[i], i
            )
        return self.scores

    def codebleu_confidence(
        self, response: str, sampled_responses: List[str], ind_: int
    ) -> float:
        """
        Calculate CodeBLEU confidence for a list of code strings.
        """
        if not sampled_responses:
            return float("nan")

        tmp_scores = []
        for candidate in sampled_responses:
            score = self.codebleu_pair(response, candidate, lang=self.lang)
            self.pair_scores[ind_].append(score)
            if not math.isnan(score):
                tmp_scores.append(score)

        return float("nan") if not tmp_scores else sum(tmp_scores) / len(tmp_scores)

    def codebleu_pair(self, response: str, candidate: str, lang: str = "python") -> float:
        """
        Calculate CodeBLEU score for a pair of code strings.
        """
        if not response or not candidate:
            return float("nan")

        try:
            res = self.calc_codebleu(
                [str(response)],
                [candidate],
                lang=lang,
                weights=(0.25, 0.25, 0.25, 0.25),
                tokenizer=None,
            )
            return float(res["codebleu"])
        except Exception as e:
            print(e)
            return float("nan")
