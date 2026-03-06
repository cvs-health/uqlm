from typing import List, Any, Optional
import asyncio
import pandas as pd
import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage
from uqlm.utils.prompts.codegen import PYTHON_JAVA_SYSTEM_PROMPT, SQL_SYSTEM_PROMPT

class CodeEquivalence:
    def __init__(self, llm: Any, system_prompt: Optional[str] = None, retries: int = 5, language="python"):
        self.llm = llm
        self.system_prompt = system_prompt
        self.language = language
        if not self.system_prompt:
            if self.language in ["python", "java"]:
                self.system_prompt = PYTHON_JAVA_SYSTEM_PROMPT.format(language=self.language)
            elif self.language == "sql":
                self.system_prompt = SQL_SYSTEM_PROMPT
            else:
                raise ValueError("language must be one of python, java, sql")
        self.retries = retries
        self.indicators, self.scores = None, None

    async def score(
        self, responses: List[str], sampled_responses: List[List[str]]
    ) -> List[List[float]]:
        if len(responses) == 0 or len(sampled_responses) == 0:
            raise ValueError("Either responses or sampled responses is empty")
        n_prompts, n_samples = len(responses), len(sampled_responses[0])
        self.scores = [[None for _ in range(n_samples)] for _ in range(n_prompts)]
        self.equivalence_cache = {}
        indices = []
        pairs = []
        for i in range(n_prompts):
            for j in range(n_samples):
                pairs.append([responses[i], sampled_responses[i][j]])
                indices.append((i, j))
        scores = await self.get_equivalence_responses(pairs)
        scores_df = pd.DataFrame({"pair": pairs, "scores": scores}, index=indices)

        retry = 0
        while retry <= self.retries:
            retry += 1

            score_failures = scores_df[pd.isna(scores_df.scores)]
            if len(score_failures) > 0:
                failure_indices = set(score_failures.index)

                tasks_tmp = [self._generate_with_identical_skip(pair) for pair in list(scores_df.loc[list(failure_indices)]["pair"])]
                retry_data = await asyncio.gather(*tasks_tmp)

                scores_df.loc[list(failure_indices), "scores"] = retry_data

            if len(score_failures) == 0:
                break

        for i, j in scores_df.index:
            self.scores[i][j] = scores_df["scores"][(i, j)]
        return self.scores

    async def _generate_with_identical_skip(self, pair: List[str]) -> float:
        code_a = str(pair[0]).strip()
        code_b = str(pair[1]).strip()
        if code_a == code_b:
            return 1.0
        
        key = code_a + "_*|\n|*_" + code_b
        rev_key = code_b + "_*|\n|*_" + code_a

        if key in self.equivalence_cache:
            return self.equivalence_cache[key]
        if rev_key in self.equivalence_cache:
            return self.equivalence_cache[rev_key]

        prompt = self.build_user_prompt(code_a, code_b)
        generation = await self.llm.ainvoke([SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)])
        score = self.normalize_verdict(getattr(generation, "content", ""))
        self.equivalence_cache[key] = score
        return float(score)
            
    async def get_equivalence_responses(self, pairs: List[List[str]]) -> List[float]:
        tasks = [self._generate_with_identical_skip(pair) for pair in pairs]
        scores = await asyncio.gather(*tasks)
        return [float(score) for score in scores]

    @staticmethod
    def build_user_prompt(code_a: str, code_b: str) -> str:
        return f"Code A:\n{code_a}\n\nCode B:\n{code_b}\n"

    @staticmethod
    def normalize_verdict(text: str) -> int:
        if not isinstance(text, str):
            return np.nan
        t = text.strip().lower().replace("-", " ")
        if "not equivalent" in t:
            return 0.0
        elif "equivalent" in t:
            return 1.0
        if any(phrase in t for phrase in ["are the same", "behave the same", "identical", "same output"]):
            return 1.0
        if any(phrase in t for phrase in ["are different", "behave differently", "not the same", "different output"]):
            return 0.0
        return np.nan