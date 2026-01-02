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


from typing import Any, List, Optional
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier
from uqlm.judges.judge import LLMJudge

DEFAULT_BLACK_BOX_SCORERS = ["semantic_negentropy", "noncontradiction", "exact_match", "cosine_sim"]
BLACK_BOX_SCORERS = DEFAULT_BLACK_BOX_SCORERS + ["bert_score", "entailment", "semantic_sets_confidence"]
DEFAULT_WHITE_BOX_SCORERS = ["normalized_probability", "min_probability"]


class ShortFormUQ(UncertaintyQuantifier):
    def __init__(self, llm: Any = None, device: Any = None, system_prompt: Optional[str] = None, max_calls_per_min: Optional[int] = None, use_n_param: bool = False, postprocessor: Optional[Any] = None) -> None:
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

        system_prompt : str, default=None
            Optional argument for user to provide custom system prompt. If prompts are list of strings and system_prompt is None,
            defaults to "You are a helpful assistant."

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
        super().__init__(llm=llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param, postprocessor=postprocessor)
        self.black_box_names = BLACK_BOX_SCORERS
        self.white_box_names = DEFAULT_WHITE_BOX_SCORERS
        self.default_black_box_names = DEFAULT_BLACK_BOX_SCORERS

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

    def _update_best(self, best_responses: List[str], include_logprobs: bool = True) -> None:
        """Updates best"""
        self.original_responses = self.responses.copy()
        for i, response in enumerate(self.responses):
            all_candidates = [response] + self.sampled_responses[i]
            index_of_best = all_candidates.index(best_responses[i])

            all_candidates.remove(best_responses[i])
            self.responses[i] = best_responses[i]
            self.sampled_responses[i] = all_candidates

            if include_logprobs:
                all_logprobs = [self.logprobs[i]] + self.multiple_logprobs[i]
                best_logprobs = all_logprobs[index_of_best]
                all_logprobs.remove(best_logprobs)
                self.logprobs[i] = best_logprobs
                self.multiple_logprobs[i] = all_logprobs

            if self.postprocessor:
                all_raw_candidates = [self.raw_responses[i]] + self.raw_sampled_responses[i]
                best_raw_response = all_raw_candidates[index_of_best]
                all_raw_candidates.remove(best_raw_response)
                self.raw_responses[i] = best_raw_response
                self.raw_sampled_responses[i] = all_raw_candidates

    def _construct_black_box_return_data(self):
        """Helper function to prepare black box return data"""
        data_to_return = {"responses": self.responses, "sampled_responses": self.sampled_responses}
        if self.postprocessor:
            if self.return_responses == "all":
                data_to_return["raw_responses"] = self.raw_responses
                data_to_return["raw_sampled_responses"] = self.raw_sampled_responses
            elif self.return_responses == "raw":
                data_to_return["responses"] = self.raw_responses
                data_to_return["sampled_responses"] = self.raw_sampled_responses

        if self.prompts:
            data_to_return["prompts"] = self.prompts

        return data_to_return

    def _display_optimization_header(self, show_progress_bars: bool) -> None:
        """Displays optimization header"""
        if show_progress_bars and self.progress_bar:
            try:
                self.progress_bar.add_task("")
                self.progress_bar.add_task("⚙️ Optimization")
            except (AttributeError, RuntimeError, OSError):
                # If progress bar fails, just continue without it
                pass
