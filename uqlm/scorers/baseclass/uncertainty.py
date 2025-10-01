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
from typing import Any, List, Optional, Union
from langchain_core.messages import BaseMessage
from rich.progress import Progress, TextColumn

from uqlm.utils.response_generator import ResponseGenerator
from uqlm.black_box.nli import NLIScorer
from uqlm.judges.judge import LLMJudge
from uqlm.utils.display import ConditionalBarColumn, ConditionalTimeElapsedColumn, ConditionalTextColumn, ConditionalSpinnerColumn

DEFAULT_BLACK_BOX_SCORERS = ["semantic_negentropy", "noncontradiction", "exact_match", "cosine_sim"]

BLACK_BOX_SCORERS = DEFAULT_BLACK_BOX_SCORERS + ["bert_score"]

WHITE_BOX_SCORERS = ["normalized_probability", "min_probability"]


class UncertaintyQuantifier:
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
        self.llm = llm
        self.device = device
        self.postprocessor = postprocessor
        self.system_prompt = system_prompt
        self.max_calls_per_min = max_calls_per_min
        self.use_n_param = use_n_param
        self.black_box_names = BLACK_BOX_SCORERS
        self.white_box_names = WHITE_BOX_SCORERS
        self.default_black_box_names = DEFAULT_BLACK_BOX_SCORERS
        self.progress_bar = None
        self.raw_responses = None
        self.raw_sampled_responses = None

    async def generate_original_responses(self, prompts: List[Union[str, List[BaseMessage]]], progress_bar: Optional[Progress] = None) -> List[str]:
        """
        This method generates original responses for uncertainty
        estimation. If specified in the child class, all responses are postprocessed
        using the callable function defined by the user.

        Parameters
        ----------
        prompts : List[Union[str, List[BaseMessage]]]
            List of prompts from which LLM responses will be generated. Prompts in list may be strings or lists of BaseMessage. If providing
            input type List[List[BaseMessage]], refer to https://python.langchain.com/docs/concepts/messages/#langchain-messages for support.

        progress_bar : rich.progress.Progress, default=None
            A progress bar object to display progress.

        Returns
        -------
        list of str
            A list of original responses for each prompt.
        """
        generations = await self._generate_responses(prompts, count=1, progress_bar=progress_bar)
        responses = generations["responses"]
        self.logprobs = generations["logprobs"]
        if self.postprocessor:
            self.raw_responses = responses
            responses = [self.postprocessor(r) for r in responses]
        return responses

    async def generate_candidate_responses(self, prompts: List[Union[str, List[BaseMessage]]], num_responses: int = 5, progress_bar: Optional[Progress] = None) -> List[List[str]]:
        """
        This method generates multiple responses for uncertainty
        estimation. If specified in the child class, all responses are postprocessed
        using the callable function defined by the user.

        Parameters
        ----------
        prompts : List[Union[str, List[BaseMessage]]]
            List of prompts from which LLM responses will be generated. Prompts in list may be strings or lists of BaseMessage. If providing
            input type List[List[BaseMessage]], refer to https://python.langchain.com/docs/concepts/messages/#langchain-messages for support.

        num_responses : int, default=5
            The number of sampled responses used to compute consistency.

        progress_bar : rich.progress.Progress, default=None
            A progress bar object to display progress.

        Returns
        -------
        list of list of str
            A list of sampled responses for each prompt.
        """
        llm_temperature = self.llm.temperature
        generations = await self._generate_responses(prompts=prompts, count=num_responses, temperature=self.sampling_temperature, progress_bar=progress_bar)
        tmp_mr, tmp_lp = generations["responses"], generations["logprobs"]
        sampled_responses, self.multiple_logprobs = [], []
        for i in range(len(prompts)):
            sampled_responses.append(tmp_mr[i * num_responses : (i + 1) * num_responses])
            if len(tmp_lp) == len(tmp_mr):
                self.multiple_logprobs.append(tmp_lp[i * num_responses : (i + 1) * num_responses])
        if self.postprocessor:
            self.raw_sampled_responses = sampled_responses
            sampled_responses = [[self.postprocessor(r) for r in m] for m in sampled_responses]
        self.llm.temperature = llm_temperature
        return sampled_responses

    async def _generate_responses(self, prompts: List[Union[str, List[BaseMessage]]], count: int, temperature: float = None, progress_bar: Optional[Progress] = None) -> List[str]:
        """Helper function to generate responses with LLM"""
        try:
            if self.llm is None:
                raise ValueError("""llm must be provided to generate responses.""")
            llm_temperature = self.llm.temperature
            if temperature:
                self.llm.temperature = temperature
            generator_object = ResponseGenerator(llm=self.llm, max_calls_per_min=self.max_calls_per_min, use_n_param=self.use_n_param)
            with contextlib.redirect_stdout(io.StringIO()):
                generations = await generator_object.generate_responses(prompts=prompts, count=count, system_prompt=self.system_prompt, progress_bar=progress_bar)
            self.llm.temperature = llm_temperature
        except Exception:
            if progress_bar:
                progress_bar.stop()
            raise
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

    def _construct_progress_bar(self, show_progress_bars: bool, _existing_progress_bar: Any = None) -> None:
        """Constructs and starts progress bar"""
        # Clean up any existing progress bar before creating/using a new one
        # Note: We don't try to reuse progress bars because after interruption in notebooks,
        # they can be in a broken state where is_started=True but they're not actually displaying
        if self.progress_bar is not None:
            try:
                self.progress_bar.stop()
            except (AttributeError, RuntimeError, OSError):
                pass
            try:
                if hasattr(self.progress_bar, "live") and self.progress_bar.live is not None:
                    self.progress_bar.live.stop()
            except (AttributeError, RuntimeError, OSError):
                pass
            # Always reset after cleanup attempt
            self.progress_bar = None

        # Handle externally provided progress bar
        if _existing_progress_bar:
            self.progress_bar = _existing_progress_bar
            try:
                self.progress_bar.start()
            except (AttributeError, RuntimeError, OSError):
                # If starting fails, fall through to create a new one
                self.progress_bar = None

        # Create a new progress bar if needed
        if show_progress_bars and not self.progress_bar:
            try:
                completion_text = "[progress.percentage]{task.completed}/{task.total}"
                self.progress_bar = Progress(ConditionalSpinnerColumn(), TextColumn("[progress.description]{task.description}"), ConditionalBarColumn(), ConditionalTextColumn(completion_text), ConditionalTimeElapsedColumn())
                self.progress_bar.start()
            except Exception as e:
                # If progress bar creation fails, continue without it
                print(f"Could not create progress bar: {e}")
                self.progress_bar = None

    def _display_generation_header(self, show_progress_bars: bool, white_box: bool = False) -> None:
        """Displays generation header"""
        if show_progress_bars and self.progress_bar:
            try:
                display_text = "🤖 Generation" if not white_box else "🤖📈 Generation & Scoring"
                self.progress_bar.add_task(display_text)
            except (AttributeError, RuntimeError, OSError):
                # If progress bar fails, just continue without it
                pass

    def _display_scoring_header(self, show_progress_bars: bool) -> None:
        """Displays scoring header"""
        if show_progress_bars and self.progress_bar:
            try:
                self.progress_bar.add_task("")
                self.progress_bar.add_task("📈 Scoring")
            except (AttributeError, RuntimeError, OSError):
                # If progress bar fails, just continue without it
                pass

    def _display_optimization_header(self, show_progress_bars: bool) -> None:
        """Displays optimization header"""
        if show_progress_bars and self.progress_bar:
            try:
                self.progress_bar.add_task("")
                self.progress_bar.add_task("⚙️ Optimization")
            except (AttributeError, RuntimeError, OSError):
                # If progress bar fails, just continue without it
                pass

    def _stop_progress_bar(self, _existing_progress_bar: Any = None) -> None:
        """Stop progress bar"""
        if self.progress_bar is not None:
            try:
                self.progress_bar.stop()
            except (AttributeError, RuntimeError, OSError):
                # If progress bar fails, just continue without it
                pass
            # Also ensure the live display is cleaned up
            try:
                if hasattr(self.progress_bar, "live") and self.progress_bar.live is not None:
                    self.progress_bar.live.stop()
            except (AttributeError, RuntimeError, OSError):
                pass
        if not _existing_progress_bar:
            self.progress_bar = None

    def _start_progress_bar(self) -> None:
        """Start progress bar"""
        if self.progress_bar is not None:
            try:
                self.progress_bar.start()
            except (AttributeError, RuntimeError, OSError):
                # If progress bar fails, just continue without it
                pass
