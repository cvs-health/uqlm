import asyncio
import time
import pandas as pd
from typing import Any, List, Tuple, Optional
import numpy as np
from rich.progress import Progress
from langchain_core.messages import SystemMessage, HumanMessage
from uqlm.utils.prompts.codegen import PYTHON_JAVA_SYSTEM_PROMPT, SQL_SYSTEM_PROMPT


class CodeClusterer:
    def __init__(self, llm: Any, system_prompt: Optional[str] = None, language: str = "python", retries: int = 5):
        """
        Class for clustering code responses.

        Parameters
        ----------
        llm : Any
            A langchain llm object to get passed to chain constructor. This is used for CodeEquivalence and FunctionalEntropy scorers. User is responsible for specifying
            temperature and other relevant parameters to the constructor of their `llm` object.

        system_prompt : Optional[str], default=None
            Optional argument for user to provide custom system prompt. If prompts are list of strings and system_prompt is None,
            defaults to "You are a helpful assistant."

        language : str, default="python"
            Specifies the language of the code. Must be one of python, java, sql.

        retries : int, default=5
            Specifies the number of retries to make if the equivalence score is not found.
        """
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
        self.progress_bar = None

    async def evaluate(self, responses: List[str], sampled_responses: List[List[str]], progress_bar: Optional[Progress] = None) -> Tuple[List[List[List[int]]], List[List[float]]]:
        """
        Evaluate the cluster of responses.

        Parameters
        ----------
        responses : List[str]
            List of responses to cluster.
        sampled_responses : List[List[str]]
            List of sampled responses to cluster.
        progress_bar : Progress, default=None
            A progress bar to display the progress of the clustering.

        Returns
        -------
        Tuple[List[List[List[int]]], List[List[float]]]
            A tuple containing the cluster indices and the original equivalence scores.
        """
        n_prompts = len(responses)
        n_samples = len(sampled_responses[0])

        if progress_bar:
            progress_task = progress_bar.add_task("  - Scoring responses with semantic sets...", total=n_prompts)
            clustered_count = [0] * n_prompts  # Track clustered responses per prompt
            prompt_completed = [False] * n_prompts

        cluster_indices = [[[0]] for _ in range(n_prompts)]
        not_yet_clustered_indices = [[j for j in range(1, n_samples + 1)] for _ in range(n_prompts)]

        def mark_clustered(prompt_idx):
            if progress_bar:
                clustered_count[prompt_idx] += 1
                if clustered_count[prompt_idx] == n_samples and not prompt_completed[prompt_idx]:
                    progress_bar.update(progress_task, advance=1)
                    prompt_completed[prompt_idx] = True

        # Round 1: Compare all anchors against all their samples
        round1_scores = await self.get_equivalence_scores(responses=responses, sampled_responses=sampled_responses)
        for i in range(n_prompts):
            for j in range(n_samples):
                if round1_scores[i][j]:
                    cluster_indices[i][0].append(j + 1)  # +1 because anchor is index 0
                    not_yet_clustered_indices[i].remove(j + 1)
                    mark_clustered(i)

        # Round 2+: Iteratively cluster remaining responses
        while any(not_yet_clustered_indices):
            # Build new cluster anchors from first non-clustered response in each row
            print("new round")
            new_anchor_indices = []
            for i in range(n_prompts):
                if not_yet_clustered_indices[i]:
                    new_anchor_idx = not_yet_clustered_indices[i][0]
                    cluster_indices[i].append([new_anchor_idx])
                    not_yet_clustered_indices[i].remove(new_anchor_idx)
                    new_anchor_indices.append((i, new_anchor_idx))
                    mark_clustered(i)
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
                        mark_clustered(prompt_idx)

        time.sleep(0.2)

        return {"cluster_indices": cluster_indices, "original_equivalence_scores": round1_scores}

    async def get_equivalence_scores(self, responses: List[str], sampled_responses: List[List[str]], progress_bar: Optional[Progress] = None) -> List[List[float]]:
        """
        Get the equivalence scores for the responses.

        Parameters
        ----------
        responses : List[str]
            List of responses to get equivalence scores for.
        sampled_responses : List[List[str]]
            List of sampled responses to get equivalence scores for.

        Returns
        -------
        List[List[float]]
            A list of lists of equivalence scores.
        """
        if len(responses) == 0 or len(sampled_responses) == 0:
            raise ValueError("Either responses or sampled responses is empty")

        if progress_bar:
            self.rows_scored = 0
            self.num_samples = len(sampled_responses[0])
            self.equivalence_task = progress_bar.add_task("  - Scoring responses with functional equivalence...", total=len(responses))

        n_prompts = len(responses)
        self.scores = [[None for _ in range(len(sampled_responses[i]))] for i in range(n_prompts)]
        self.equivalence_cache = {}
        indices = []
        pairs = []
        for i in range(n_prompts):
            for j in range(len(sampled_responses[i])):
                pairs.append([responses[i], sampled_responses[i][j]])
                indices.append((i, j))
        scores = await self._get_equivalence_responses(pairs, progress_bar=progress_bar)
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

    async def _generate_with_identical_skip(self, pair: List[str], progress_bar: Optional[Progress] = None) -> float:
        """
        Generate the equivalence score for a pair of responses.

        Parameters
        ----------
        pair : List[str]
            A pair of responses to generate the equivalence score for.

        Returns
        -------
        float
            The equivalence score for the pair of responses.
        """
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
        if progress_bar:
            self.rows_scored += 1
            if self.rows_scored % self.num_samples == 0:
                progress_bar.update(self.equivalence_task, advance=1)
        return float(score)

    async def _get_equivalence_responses(self, pairs: List[List[str]], progress_bar: Optional[Progress] = None) -> List[float]:
        """
        Get the equivalence scores for a list of pairs of responses.

        Parameters
        ----------
        pairs : List[List[str]]
            A list of pairs of responses to get equivalence scores for.

        Returns
        -------
        List[float]
            A list of equivalence scores for the pairs of responses.
        """
        tasks = [self._generate_with_identical_skip(pair, progress_bar=progress_bar) for pair in pairs]
        scores = await asyncio.gather(*tasks)
        return [float(score) for score in scores]

    @staticmethod
    def build_user_prompt(code_a: str, code_b: str) -> str:
        """
        Build the user prompt for the equivalence score.

        Parameters
        ----------
        code_a : str
            The first code to compare.
        code_b : str
            The second code to compare.

        Returns
        -------
        str
            The user prompt for the equivalence score.
        """
        return f"Code A:\n{code_a}\n\nCode B:\n{code_b}\n"

    @staticmethod
    def normalize_verdict(text: str) -> int:
        """
        Normalize the verdict for the equivalence score.

        Parameters
        ----------
        text : str
            The text to normalize.

        Returns
        -------
        int
            The normalized verdict for the equivalence score.
        """
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
