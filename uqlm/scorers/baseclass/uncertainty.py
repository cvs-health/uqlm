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


import io
import contextlib
import pandas as pd
from typing import Any, Dict, List, Optional
from uqlm.utils.response_generator import ResponseGenerator
from uqlm.black_box.nli import NLIScorer
from uqlm.judges.judge import LLMJudge

DEFAULT_BLACK_BOX_SCORERS = ["semantic_negentropy", "noncontradiction", "exact_match", "cosine_sim"]

BLACK_BOX_SCORERS = DEFAULT_BLACK_BOX_SCORERS + ["bert_score", "bleurt"]

WHITE_BOX_SCORERS = ["normalized_probability", "min_probability"]


class UncertaintyQuantifier:
    def __init__(self, llm: Any = None, device: Any = None, system_prompt: str = "You are a helpful assistant", max_calls_per_min: Optional[int] = None, use_n_param: bool = False, postprocessor: Optional[Any] = None) -> None:
        """
        Parent class for uncertainty quantification of LLM responses

        Parameters
        ----------
        llm : BaseChatModel
            A langchain llm object to get passed to chain constructor. User is responsible for specifying
            temperature and other relevant parameters to the constructor of their `llm` object.

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. Only applies to 'semantic_negentropy', 'noncontradiction'
            scorers. Pass a torch.device to leverage GPU.

        system_prompt : str or None, default="You are a helpful assistant."
            Optional argument for user to provide custom system prompt

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.

        use_n_param : bool, default=False
            Specifies whether to use `n` parameter for `BaseChatModel`. Not compatible with all
            `BaseChatModel` classes. If used, it speeds up the generation process substantially when num_responses > 1.

        postprocessor : callable, default=None
            A user-defined function that takes a string input and returns a string. Used for postprocessing
            outputs.
        """
        self.llm = llm
        self.device = device
        self.postprocessor = postprocessor
        self.system_prompt = system_prompt
        self.max_calls_per_min = max_calls_per_min
        self.use_n_param = use_n_param
        self.black_box_names = BLACK_BOX_SCORERS
        self.white_box_names = WHITE_BOX_SCORERS
        self.default_black_box_names = DEFAULT_BLACK_BOX_SCORERS

    async def generate_original_responses(self, prompts: List[str]) -> List[str]:
        """
        This method generates original responses for uncertainty
        estimation. If specified in the child class, all responses are postprocessed
        using the callable function defined by the user.
        """
        print("Generating responses...")
        generations = await self._generate_responses(prompts, count=1)
        responses = generations["responses"]
        self.logprobs = generations["logprobs"]
        if self.postprocessor:
            responses = [self.postprocessor(r) for r in responses]
        return responses

    async def generate_candidate_responses(self, prompts: List[str]) -> List[List[str]]:
        """
        This method generates multiple responses for uncertainty
        estimation. If specified in the child class, all responses are postprocessed
        using the callable function defined by the user.
        """
        llm_temperature = self.llm.temperature
        print("Generating candidate responses...")
        generations = await self._generate_responses(prompts=prompts, count=self.num_responses, temperature=self.sampling_temperature)
        tmp_mr, tmp_lp = generations["responses"], generations["logprobs"]
        sampled_responses, self.multiple_logprobs = [], []
        for i in range(len(prompts)):
            sampled_responses.append(tmp_mr[i * self.num_responses : (i + 1) * self.num_responses])
            if len(tmp_lp) == len(tmp_mr):
                self.multiple_logprobs.append(tmp_lp[i * self.num_responses : (i + 1) * self.num_responses])
        if self.postprocessor:
            sampled_responses = [[self.postprocessor(r) for r in m] for m in sampled_responses]
        self.llm.temperature = llm_temperature
        return sampled_responses

    async def _generate_responses(self, prompts: List[str], count: int, temperature: float = None) -> List[str]:
        """Helper function to generate responses with LLM"""
        if self.llm is None:
            raise ValueError("""llm must be provided to generate responses.""")
        llm_temperature = self.llm.temperature
        if temperature:
            self.llm.temperature = temperature
        generator_object = ResponseGenerator(llm=self.llm, max_calls_per_min=self.max_calls_per_min, use_n_param=self.use_n_param)
        with contextlib.redirect_stdout(io.StringIO()):
            generations = await generator_object.generate_responses(prompts=prompts, count=count, system_prompt=self.system_prompt)
        self.llm.temperature = llm_temperature
        return {"responses": generations["data"]["response"], "logprobs": generations["metadata"]["logprobs"]}

    def _construct_judge(self, llm: Any = None) -> LLMJudge:
        """
        Constructs LLMJudge object
        """
        if llm is None:
            llm_temperature = self.llm.temperature
            self.llm.temperature = 0
            self_judge = LLMJudge(llm=self.llm, max_calls_per_min=self.max_calls_per_min)
            self.llm.temperature = llm_temperature
            return self_judge
        else:
            return LLMJudge(llm=llm)

    def _setup_nli(self, nli_model_name: Any) -> None:
        """Set up NLI scorer"""
        self.nli_scorer = NLIScorer(nli_model_name=self.nli_model_name, device=self.device, max_length=self.max_length, verbose=self.verbose)

    def _update_best(self, best_responses: List[str]) -> None:
        """Updates best"""
        self.original_responses = self.responses.copy()
        for i, response in enumerate(self.responses):
            all_candidates = [response] + self.sampled_responses[i]
            all_logprobs = [self.logprobs[i]] + self.multiple_logprobs[i]
            best_logprobs = all_logprobs[all_candidates.index(best_responses[i])]

            all_candidates.remove(best_responses[i])
            self.responses[i] = best_responses[i]
            self.sampled_responses[i] = all_candidates

            all_logprobs.remove(best_logprobs)
            self.logprobs[i] = best_logprobs
            self.multiple_logprobs[i] = all_logprobs


class UQResult:
    def __init__(self, result: Dict[str, Any]) -> None:
        """
        Class that characterizes result of an UncertaintyQuantifier.

        Parameters
        ----------
        result: dict
            A dictionary that is defined during `evaluate` or `tune_params` method
        """
        self.data = result.get("data")
        self.metadata = result.get("metadata")
        self.parameters = result.get("parameters")
        self.confidence_scores = self.data.get("confidence_scores")
        self.responses = self.data.get("responses")
        self.sampled_responses = None if not self.data.get("responses") else self.data.get("responses")
        self.result_dict = result

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns result in dictionary form
        """
        return self.result_dict

    def to_df(self) -> pd.DataFrame:
        """
        Returns result in pd.DataFrame
        """
        rename_dict = {col: col[:-1] for col in self.result_dict["data"].keys() if col.endswith("s") and col != "sampled_responses"}

        return pd.DataFrame(self.result_dict["data"]).rename(columns=rename_dict)
