import asyncio
import time
import pandas as pd
from typing import Any, List, Tuple, Optional
import numpy as np
from rich.progress import Progress
from langchain_core.messages import SystemMessage, HumanMessage

PYTHON_JAVA_SYSTEM_PROMPT = """
You are a {language} code equivalence judge.

Definition:
Two {language} code blocks are considered functionally equivalent if they would produce the same outputs for the same inputs.

Consider equivalent:
- Different implementations or algorithms that achieve the same result
- Refactored or restructured code with the same behavior
- Minor variations in edge case handling

Consider NOT equivalent:
- Code that would produce different outputs for the same inputs
- Code where one is incomplete or missing functionality present in the other

Decision rule:
- If both code snippets would generally produce the same results → output exactly: "Equivalent"
- If the code snippets would produce different outputs → output exactly: "Not Equivalent"

Output format:
Output EXACTLY one of: "Equivalent" OR "Not Equivalent".
Do not add explanations, reasoning, punctuation, or extra text.
"""

SQL_SYSTEM_PROMPT = """
You are a SQLite query equivalence judge.

Definition:
Two SQLite queries are considered semantically equivalent only if, when executed against the same database state, they produce exactly the same result set.

Same result set means:
- The same rows (treating rows as unordered sets, unless ORDER BY is specified in both queries)
- The same column values in each row
- The same column order

Ignore:
- Purely syntactic differences (formatting, whitespace, capitalization of keywords)
- Use of aliases that do not affect the result set
- Equivalent expressions (e.g., `WHERE a = 1 AND b = 2` vs. `WHERE b = 2 AND a = 1`)
- Different join syntax with equivalent semantics (e.g., implicit vs. explicit JOIN)
- Use of parentheses that do not change query semantics
- Comments

Do NOT ignore:
- Different row ordering if either query specifies ORDER BY
- NULL handling differences that affect results
- DISTINCT vs. non-DISTINCT if it changes output rows
- Column order differences in the SELECT clause

Decision rule:
- If both queries would return the same result set on any valid database state → output exactly: "Equivalent"
- If any valid database state exists where the queries would return different results → output exactly: "Not Equivalent"

Output format: 
Output EXACTLY one of: "Equivalent" OR "Not Equivalent".
Do not add explanations, reasoning, punctuation, or extra text.
"""


class CodeClusterer:
    def __init__(self, llm: Any, system_prompt: Optional[str] = None, length_normalize: bool = False, language: str = "python", retries: int = 5):
        self.llm = llm
        self.system_prompt = system_prompt
        self.length_normalize = length_normalize
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
        self.progress_bar = None

    async def get_equivalence_scores(self, responses: List[str], sampled_responses: List[List[str]]) -> List[List[float]]:
        if len(responses) == 0 or len(sampled_responses) == 0:
            raise ValueError("Either responses or sampled responses is empty")

        n_prompts = len(responses)
        self.scores = [[None for _ in range(len(sampled_responses[i]))] for i in range(n_prompts)]
        self.equivalence_cache = {}
        indices = []
        pairs = []
        for i in range(n_prompts):
            for j in range(len(sampled_responses[i])):
                pairs.append([responses[i], sampled_responses[i][j]])
                indices.append((i, j))
        scores = await self._get_equivalence_responses(pairs)
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

    async def evaluate(self, responses: List[str], sampled_responses: List[List[str]], progress_bar: Optional[Progress] = None) -> Tuple[List[List[List[int]]], List[List[float]]]:
        n_prompts = len(responses)
        n_samples = len(sampled_responses[0])

        if progress_bar:
            progress_task = progress_bar.add_task("  - Scoring responses with semantic sets...", total=len(responses))
            rows_completed = [False] * n_prompts

        # Initialize: each prompt starts with cluster containing just the anchor (index 0)
        cluster_indices = [[[0]] for _ in range(n_prompts)]
        not_yet_clustered_indices = [[j for j in range(1, n_samples + 1)] for _ in range(n_prompts)]

        # Round 1: Compare all anchors against all their samples
        round1_scores = await self.get_equivalence_scores(responses=responses, sampled_responses=sampled_responses)
        for i in range(n_prompts):
            for j in range(n_samples):
                if round1_scores[i][j]:
                    cluster_indices[i][0].append(j + 1)  # +1 because anchor is index 0
                    not_yet_clustered_indices[i].remove(j + 1)
            if progress_bar and not not_yet_clustered_indices[i]:
                progress_bar.update(progress_task, advance=1)
                rows_completed[i] = True

        # Round 2+: Iteratively cluster remaining responses
        while any(not_yet_clustered_indices):
            # Build new cluster anchors from first non-clustered response in each row
            new_anchor_indices = []
            for i in range(n_prompts):
                if not_yet_clustered_indices[i]:
                    new_anchor_idx = not_yet_clustered_indices[i][0]
                    cluster_indices[i].append([new_anchor_idx])
                    not_yet_clustered_indices[i].remove(new_anchor_idx)
                    new_anchor_indices.append((i, new_anchor_idx))
                else:
                    new_anchor_indices.append(None)

            # Compare new anchors against remaining non-clustered responses
            responses_tmp = []
            sampled_responses_tmp = []
            prompt_mapping = []  # Maps tmp index back to (prompt_idx, new_cluster_idx)

            for i in range(n_prompts):
                if new_anchor_indices[i] is None or not not_yet_clustered_indices[i]:
                    continue
                _, new_anchor_idx = new_anchor_indices[i]
                # Get the actual response text for the new anchor
                all_responses_i = [responses[i]] + list(sampled_responses[i])
                new_anchor = all_responses_i[new_anchor_idx]

                # Remaining responses to compare against
                remaining = [all_responses_i[idx] for idx in not_yet_clustered_indices[i]]

                responses_tmp.append(new_anchor)
                sampled_responses_tmp.append(remaining)
                prompt_mapping.append((i, len(cluster_indices[i]) - 1, list(not_yet_clustered_indices[i])))

            if not responses_tmp:
                break

            # Get equivalence scores for this round
            round_scores = await self.get_equivalence_scores(responses=responses_tmp, sampled_responses=sampled_responses_tmp)

            # Assign matches to clusters
            for tmp_idx, (prompt_idx, cluster_idx, remaining_indices) in enumerate(prompt_mapping):
                for j, orig_idx in enumerate(remaining_indices):
                    if round_scores[tmp_idx][j]:
                        cluster_indices[prompt_idx][cluster_idx].append(orig_idx)
                        not_yet_clustered_indices[prompt_idx].remove(orig_idx)

            if progress_bar:
                self._progress_update_loop(progress_bar, progress_task, not_yet_clustered_indices, rows_completed, n_prompts)

        if progress_bar:
            self._progress_update_loop(progress_bar, progress_task, not_yet_clustered_indices, rows_completed, n_prompts)

        time.sleep(0.2)

        return {"cluster_indices": cluster_indices, "original_equivalence_scores": round1_scores}

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

    async def _get_equivalence_responses(self, pairs: List[List[str]]) -> List[float]:
        tasks = [self._generate_with_identical_skip(pair) for pair in pairs]
        scores = await asyncio.gather(*tasks)
        return [float(score) for score in scores]

    @staticmethod
    def _progress_update_loop(progress_bar, progress_task, not_yet_clustered_indices, rows_completed, n_prompts):
        for i in range(n_prompts):
            if not not_yet_clustered_indices[i] and not rows_completed[i]:
                progress_bar.update(progress_task, advance=1)
                rows_completed[i] = True

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
