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
from rich.errors import LiveError

from uqlm.utils.response_generator import ResponseGenerator
from uqlm.nli.nli import NLI
from uqlm.utils.display import ConditionalBarColumn, ConditionalTimeElapsedColumn, ConditionalTextColumn, ConditionalSpinnerColumn


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
        self.progress_bar = None
        self.raw_responses = None
        self.raw_sampled_responses = None

    async def generate_original_responses(self, prompts: List[Union[str, List[BaseMessage]]], top_k_logprobs: Optional[int] = None, progress_bar: Optional[Progress] = None) -> List[str]:
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
        generations = await self._generate_responses(prompts, count=1, top_k_logprobs=top_k_logprobs, progress_bar=progress_bar)
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
        generations = await self._generate_responses(prompts=prompts, count=num_responses, temperature=self.sampling_temperature, top_k_logprobs=None, progress_bar=progress_bar)
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

    async def _generate_responses(self, prompts: List[Union[str, List[BaseMessage]]], count: int, temperature: float = None, top_k_logprobs: Optional[int] = None, progress_bar: Optional[Progress] = None) -> List[str]:
        """Helper function to generate responses with LLM"""
        try:
            if self.llm is None:
                raise ValueError("""llm must be provided to generate responses.""")
            llm_temperature = self.llm.temperature
            if temperature:
                self.llm.temperature = temperature
            generator_object = ResponseGenerator(llm=self.llm, max_calls_per_min=self.max_calls_per_min, use_n_param=self.use_n_param, top_k_logprobs=top_k_logprobs)
            with contextlib.redirect_stdout(io.StringIO()):
                generations = await generator_object.generate_responses(prompts=prompts, count=count, system_prompt=self.system_prompt, progress_bar=progress_bar)
            self.llm.temperature = llm_temperature
        except Exception:
            if progress_bar:
                progress_bar.stop()
            raise
        return {"responses": generations["data"]["response"], "logprobs": generations["metadata"]["logprobs"]}

    def _setup_nli(self, nli_model_name: Any) -> None:
        """Set up NLI model"""
        self.nli = NLI(nli_model_name=nli_model_name, device=self.device, max_length=self.max_length, verbose=self.verbose)

    def _construct_progress_bar(self, show_progress_bars: bool, _existing_progress_bar: Any = None) -> None:
        """Constructs and starts progress bar"""
        try:
            if _existing_progress_bar:
                self.progress_bar = _existing_progress_bar
                self.progress_bar.start()

            elif show_progress_bars and not self.progress_bar:
                completion_text = "[progress.percentage]{task.completed}/{task.total}"
                self.progress_bar = Progress(ConditionalSpinnerColumn(), TextColumn("[progress.description]{task.description}"), ConditionalBarColumn(), ConditionalTextColumn(completion_text), ConditionalTimeElapsedColumn())
                self.progress_bar.start()
        except LiveError:
            print("Could not create progress bar")
            self.progress_bar = None
            pass

    def _display_generation_header(self, show_progress_bars: bool, generation_type: str = "default") -> None:
        """Displays generation header"""
        if show_progress_bars and self.progress_bar:
            try:
                if generation_type == "default":
                    display_text = "🤖 Generation"
                elif generation_type == "white_box":
                    display_text = "🤖🧮 Generation with Logprobs"
                elif generation_type == "claim_qa":
                    display_text = "\n🤖 Claim-QA Answer Generation"
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
