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

import asyncio
import itertools
import time
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from uqlm.utils.warn import beta_warning


generator_type_to_progress_msg = {"judge": "Scoring responses with LLM-as-a-Judge", "original": "Generating responses", "p_true": "Scoring responses with P(True)", "grader": "Grading responses against provided ground truth answers", "factscore_grader": "Grading claims/sentences against Wikipedia texts", "question_generator": "Generating questions for each claim/sentence"}


class ResponseGenerator:
    def __init__(self, llm: BaseChatModel = None, max_calls_per_min: Optional[int] = None, use_n_param: bool = False, top_k_logprobs: Optional[int] = None) -> None:
        """
        Class for generating data from a provided set of prompts

        Parameters
        ----------
        llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        max_calls_per_min : int, default=None
            Used to control rate limiting

        use_n_param : bool, default=False
            Specifies whether to use `n` parameter for `BaseChatModel`. Not compatible with all
            `BaseChatModel` classes. If used, it speeds up the generation process substantially when count > 1.
        """
        self.llm = llm
        self.use_n_param = use_n_param
        self.max_calls_per_min = max_calls_per_min
        self.progress = None
        self.progress_task = None
        self.generator_type_to_progress_msg = generator_type_to_progress_msg
        self.response_generator_type = "original"
        self.top_k_logprobs = top_k_logprobs

    async def generate_responses(self, prompts: List[Union[str, List[BaseMessage]]], system_prompt: Optional[str] = None, count: int = 1, progress_bar: Optional[Progress] = None) -> Dict[str, Any]:
        """
        Generates responses from a provided set of prompts. For each prompt, `count` responses are generated.

        Parameters
        ----------
        prompts : List[Union[str, List[BaseMessage]]]
            List of prompts from which LLM responses will be generated. Prompts in list may be strings or lists of BaseMessage. If providing
            input type List[List[BaseMessage]], refer to https://python.langchain.com/docs/concepts/messages/#langchain-messages for support.

        system_prompt : str, default=None
            Optional argument for user to provide custom system prompt. If prompts are list of strings and system_prompt is None,
            defaults to "You are a helpful assistant."

        count : int, default=1
            Specifies number of responses to generate for each prompt.

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses

        Returns
        -------
        dict
            A dictionary with two keys: 'data' and 'metadata'.

            'data' : dict
                A dictionary containing the prompts and responses.

                'prompt' : list
                    A list of prompts.
                'response' : list
                    A list of responses corresponding to the prompts.

            'metadata' : dict
                A dictionary containing metadata about the generation process.

                'temperature' : float
                    The temperature parameter used in the generation process.
                'count' : int
                    The count of prompts used in the generation process.
                'system_prompt' : str
                    The system prompt used for generating responses
        """
        assert isinstance(self.llm, BaseChatModel), """
            llm must be an instance of langchain_core.language_models.chat_models.BaseChatModel
        """
        if any(isinstance(prompt, list) and all(isinstance(item, BaseMessage) for item in prompt) for prompt in prompts):
            beta_warning("Use of BaseMessage in prompts argument is in beta. Please use it with caution as it may change in future releases.")

        if self.llm.temperature == 0:
            assert count == 1, "temperature must be greater than 0 if count > 1"
        self._update_count(count)
        self.system_message = None if not system_prompt else SystemMessage(system_prompt)

        generations, duplicated_prompts = await self._generate_in_batches(prompts=prompts, progress_bar=progress_bar)

        responses = generations["responses"]
        logprobs = generations["logprobs"]
        return {"data": {"prompt": self._enforce_strings(duplicated_prompts), "response": self._enforce_strings(responses)}, "metadata": {"system_prompt": system_prompt, "temperature": self.llm.temperature, "count": self.count, "logprobs": logprobs}}

    def _create_tasks(self, prompts: List[Union[str, List[BaseMessage]]]) -> Tuple[List[Any], List[str]]:
        """
        Creates a list of async tasks and returns duplicated prompt list
        with each prompt duplicated `count` times
        """
        duplicated_prompts = [prompt for prompt, i in itertools.product(prompts, range(self.count))]
        if self.use_n_param:
            tasks = [self._async_api_call(prompt=prompt, count=self.count) for prompt in prompts]
        else:
            tasks = [self._async_api_call(prompt=prompt, count=1) for prompt in duplicated_prompts]
        return tasks, duplicated_prompts

    def _update_count(self, count: int) -> None:
        """Updates self.count parameter and self.llm as necessary"""
        self.count = count
        if self.use_n_param:
            self.llm.n = count

    async def _generate_in_batches(self, prompts: List[Union[str, List[BaseMessage]]], progress_bar: Optional[bool] = True) -> Tuple[Dict[str, List[Any]], List[str]]:
        """Executes async IO with langchain in batches to avoid rate limit error"""
        assert self.count > 0, "count must be greater than 0"
        self.progress_bar = progress_bar
        if self.max_calls_per_min:
            batch_size = max(1, self.max_calls_per_min // self.count)
            check_batch_time = True
        else:
            batch_size = len(prompts)
        prompts_partition = list(self._split(prompts, batch_size))

        duplicated_prompts = []
        generations = {"responses": [], "logprobs": []}
        if self.progress_bar:
            if self.count == 1:
                self.progress_task = self.progress_bar.add_task(f"  - {self.generator_type_to_progress_msg[self.response_generator_type]}...", total=len(prompts))
            else:
                self.progress_task = self.progress_bar.add_task(f"  - Generating candidate responses ({self.count} per prompt)...", total=len(prompts) * self.count)

        for batch_idx, prompt_batch in enumerate(prompts_partition):
            if batch_idx == len(prompts_partition) - 1:
                check_batch_time = False
            await self._process_batch(prompt_batch, duplicated_prompts, generations, check_batch_time)
        time.sleep(0.1)
        return generations, duplicated_prompts

    async def _process_batch(self, prompt_batch: List[Union[str, List[BaseMessage]]], duplicated_prompts: List[str], generations: Dict[str, List[Any]], check_batch_time: bool) -> None:
        """Process a single batch of prompts"""
        start = time.time()
        # generate responses for current batch
        tasks, duplicated_batch_prompts = self._create_tasks(prompt_batch)
        generations_batch = await asyncio.gather(*tasks)
        responses_batch, logprobs_batch = [], []
        for g in generations_batch:
            responses_batch.extend(g["responses"])
            logprobs_batch.extend(g["logprobs"])

        # extend lists to include current batch
        duplicated_prompts.extend(duplicated_batch_prompts)
        generations["responses"].extend(responses_batch)
        generations["logprobs"].extend(logprobs_batch)
        stop = time.time()

        # pause if needed
        if (stop - start < 60) and check_batch_time:
            time.sleep(61 - stop + start)

    async def _async_api_call(self, prompt: Union[str, List[BaseMessage]], count: int = 1) -> Dict[str, Any]:
        """Generates responses asynchronously using an RunnableSequence object"""
        if isinstance(prompt, str):
            system_message = SystemMessage("You are a helpful assistant.") if not self.system_message else self.system_message
            messages = [system_message, HumanMessage(prompt)]
        elif isinstance(prompt, list):
            if all(isinstance(item, BaseMessage) for item in prompt):
                messages = prompt if not self.system_message else [self.system_message] + prompt
        else:
            raise ValueError("prompts must be list of strings or list of lists of BaseMessage instances. For support with LangChain BaseMessage usage, refer here: https://python.langchain.com/docs/concepts/messages")

        if not self.top_k_logprobs:
            result = await self.llm.agenerate([messages])
            logprobs = [None] * count
            if hasattr(self.llm, "logprobs"):
                if self.llm.logprobs:
                    logprobs = self._extract_logprobs(logprobs=logprobs, result=result, count=count)
            result_dict = {"logprobs": logprobs, "responses": [result.generations[0][i].text for i in range(count)]}
        else:
            result_dict = await self.agenerate_with_top_logprobs(messages, count=count)
        if self.progress_bar:
            for _ in range(count):
                self.progress_bar.update(self.progress_task, advance=1)
        return result_dict

    async def agenerate_with_top_logprobs(self, messages: List[BaseMessage], count: int) -> Any:
        """Use agenerate method with top_logprobs configured"""
        logprobs = [None] * count
        result = None
        if "openai" in self.llm.__str__().lower():
            result = await self.llm.agenerate([messages], logprobs=True, top_logprobs=self.top_k_logprobs)
        elif "google" in self.llm.__str__().lower() or "gemini" in self.llm.__str__().lower():
            self.llm.logprobs = self.top_k_logprobs
            result = await self.llm.agenerate([messages])
        else:
            try:
                result = await self.llm.agenerate([messages], logprobs=True, top_logprobs=self.top_k_logprobs)
            except Exception:
                try:
                    self.llm.logprobs = self.top_k_logprobs
                    result = await self.llm.agenerate([messages])
                except Exception:
                    pass
        if result is None:  # if all attempts fail
            return {"logprobs": [None] * count, "responses": []}
        logprobs = self._extract_logprobs(logprobs=logprobs, result=result, count=count)
        return {"logprobs": logprobs, "responses": [result.generations[0][i].text for i in range(count)]}

    @staticmethod
    def _extract_logprobs(logprobs: Any, result: Any, count: int):
        for i in range(count):
            if "logprobs_result" in result.generations[0][i].generation_info:
                logprobs[i] = result.generations[0][i].generation_info["logprobs_result"]

            elif "logprobs" in result.generations[0][i].generation_info:
                if "content" in result.generations[0][i].generation_info["logprobs"]:
                    logprobs[i] = result.generations[0][i].generation_info["logprobs"]["content"]
            else:
                warnings.warn("Model did not provide logprobs in API response. White-box scores for this response may be set to np.nan.")
                logprobs[i] = [{"token": "UNABLE TO GET LOGPROBS", "logprob": np.nan, "top_logprobs": []}]
        return logprobs

    @staticmethod
    def _enforce_strings(texts: List[Any]) -> List[str]:
        """Enforce that all outputs are strings"""
        return [str(r) for r in texts]

    @staticmethod
    def _split(list_a: List[str], chunk_size: int) -> Iterator[List[str]]:
        """Partitions list"""
        for i in range(0, len(list_a), chunk_size):
            yield list_a[i : i + chunk_size]
