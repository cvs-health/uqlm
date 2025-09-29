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
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.utils.nli import EntailmentLLMJudge, NLI
from uqlm.utils.postprocessors import claims_postprocessor
from uqlm.blackbox.baseclass.similarity_scorer import SimilarityScorer
from uqlm.blackbox.baseclass.claims_scorer import ClaimScorer
from uqlm.utils.response_generator import ResponseGenerator
from uqlm.utils.llm_config import get_llm_name
from datetime import datetime
from datasets import load_dataset
from pathlib import Path
import logging
import json
import os


class FactScoreBenchmark:
    def __init__(self, judge_llm: BaseChatModel, entailment_method: str = "nli", data_dir: str = "~/.uqlm/benchmark_results"):
        self.judge_llm = judge_llm
        self.judge_llm_name = get_llm_name(judge_llm)
        self.entailment_method = entailment_method
        self.benchmark_data = {}
        self.data_dir = Path(data_dir)

    async def evaluate_scorers(
        self,
        llms: list[BaseChatModel] | BaseChatModel,
        scorers: list[SimilarityScorer | ClaimScorer] | SimilarityScorer | ClaimScorer,
        sampling_temperature: float = 0.7,
        num_responses: int = 4,
        save_results: bool = True,
        show_results: bool = False,  # TODO: add visualization logic
        use_cached_results: bool = True,  # TODO: add caching logic
        progress: bool = True,
    ) -> dict:
        """
        Parameters
        ----------
        llms: list[BaseChatModel] | BaseChatModel
            List of LLMs to be evaluated.
        scorers: list[SimilarityScorer | ClaimScorer] | SimilarityScorer | ClaimScorer
            List of scorers to be evaluated. Can be SimilarityScorer, ClaimScorer, or a combination of both.
        sampling_temperature: float
            The temperature to use for sampling responses.
        num_responses: int
            The number of responses to sample.
        save_results: bool | str
            Whether to save the results. If True, the results will be saved to a json file. If a string, the results will be saved to a json file with the given path.
        show_results: bool
            Whether to show tables and visualizations of the results.
        use_cached_results: bool
            Whether to use cached results.
        progress: bool
            Whether to show a progress bar.
        Returns
        -------
        dict
            Dictionary containing the results of the evaluation.
        """
        results = {}
        if isinstance(scorers, (SimilarityScorer, ClaimScorer)):
            scorers = [scorers]
        if any(not isinstance(scorer, ClaimScorer) for scorer in scorers):
            logging.warning("Some selected scorers are not ClaimScorers and thus won't have claim level scores.")

        # get factscore data
        ds = load_dataset("dskar/FActScore", split="test")
        ds = ds.to_pandas()[:2]
        # entities = ds["entity"]
        prompts = ds["factscore_prompt"]
        prompt_ids = ds.index.tolist()
        source_texts = ds["wikipedia_text"]

        run_id = round(datetime.now().timestamp())

        self.benchmark_data["metadata"] = {
            "run_id": run_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_path": list(ds.info.download_checksums.keys())[0],
            "dataset_name": ds.info.dataset_name,
            "dataset_version": ds.info.version,
            "judge_llm": self.judge_llm_name,
            "llms": [get_llm_name(llm) for llm in llms],
            "scorers": [scorer.__class__.__name__ for scorer in scorers],
            "sampling_temperature": sampling_temperature,
            "original_temperature": 0,
            "num_responses": num_responses,
        }

        self.benchmark_data["data"] = {}

        # run benchmark for each llm provided
        for llm in llms:
            llm_name = get_llm_name(llm)
            self.benchmark_data["data"][llm_name] = {}

            # get llm responses
            llm.temperature = 0
            original_responses = await self._get_responses(llm=llm, prompts=prompts, num_responses=1, progress=progress)
            llm.temperature = sampling_temperature
            sampled_responses = await self._get_responses(llm=llm, prompts=prompts, num_responses=num_responses, progress=progress)
            for i, prompt_id in enumerate(prompt_ids):
                self.benchmark_data["data"][llm_name][prompt_id] = {"original_response": original_responses[i], "sampled_responses": sampled_responses[i]}

            # Decompose responses into claims
            claims_responses = await claims_postprocessor(llm=self.judge_llm, responses=original_responses)

            for i, prompt_id in enumerate(prompt_ids):
                self.benchmark_data["data"][llm_name][prompt_id]["original_response_claims"] = claims_responses[i]

            # get entailment between original_response_claims and sampled responses
            sampled_response_entailment = await self._find_entailment(claims_responses, sampled_responses, method=self.entailment_method)

            # get entailment between original_response_claims and source wikipedia text
            source_text_entailment = await self._find_entailment(claims_responses, source_texts, method=self.entailment_method)

            for i, prompt_id in enumerate(prompt_ids):
                self.benchmark_data["data"][llm_name][prompt_id]["sampled_response_entailment"] = sampled_response_entailment[i]
                self.benchmark_data["data"][llm_name][prompt_id]["source_text_entailment"] = source_text_entailment[i]

            # get

        return results

    async def _get_responses(self, llm: BaseChatModel, prompts: list[str], num_responses: int = 4, progress: bool = True) -> list[str]:
        """
        Get responses from LLMs.
        """
        rg = ResponseGenerator(llm=llm, max_calls_per_min=250)
        generations = await rg.generate_responses(prompts=prompts, count=num_responses)
        return generations["data"]["response"]

    async def _find_entailment(self, claims: list[str] | str, source_texts: list[str] | str, method: str = "nli") -> list[dict]:
        """
        Parameters
        ----------
        claims: list[str] | str
            Claims to be evaluated.
        source_texts: list[str] | str
            Source texts to be evaluated.
        Returns
        -------
        list[dict]
            List of dictionaries containing entailment categorization for each claim.
        """
        if isinstance(claims, str):
            claims = [claims]
        if isinstance(source_texts, str):
            source_texts = [source_texts]
        results = []
        if method == "nli":
            nli = NLI()
            for claims, source_text in zip(claims, source_texts):
                tmp_res = []
                for claim in claims:
                    entailment = nli.predict(claim, source_text)
                    tmp_res.append(entailment)
                results.append(tmp_res)
        elif method == "llm":
            nli = EntailmentLLMJudge(llm=self.judge_llm)
            results = await nli.predict(claims, source_texts)
        else:
            raise ValueError(f"Invalid method: {method}")

        return results

    def _save_results(self, results: dict, run_id: str) -> None:
        """
        Save the results to a json file.
        """
        with open(f"factscore_benchmark_results_{run_id}.json", "w") as f:
            json.dump(results, f)

    def _load_results(self, run_id: str) -> dict:
        """
        Load the results from a json file.
        """
        with open(f"factscore_benchmark_results_{run_id}.json", "r") as f:
            return json.load(f)

    def _show_results(self, results: dict) -> None:
        """
        Show the results.
        """
        print(results)

    def _check_cache(self, run_id: str) -> bool:
        """
        Check if the results are cached.
        """
        return os.path.exists(f"factscore_benchmark_results_{run_id}.json")
