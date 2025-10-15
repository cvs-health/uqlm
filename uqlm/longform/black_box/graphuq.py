from uqlm.longform.black_box.baseclass.claims_scorer import ClaimScorer, ClaimScores
from typing import List, Optional
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel


class GraphUQScorer(ClaimScorer):
    def __init__(self, judge_llm: BaseChatModel) -> None:
        super().__init__(judge_llm)

    def evaluate(self, claim_sets: List[List[str]], sampled_claim_sets: List[List[List[str]]] = None, progress_bar: Optional[Progress] = None) -> ClaimScores:
        pass
