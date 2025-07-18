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
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn


class ResponseGenerator:
    def __init__(self, llm: BaseChatModel = None, max_calls_per_min: Optional[int] = None, use_n_param: bool = False) -> None:
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
        self.is_judge = False

    async def generate_responses(self, prompts: List[str], system_prompt: str = "You are a helpful assistant.", count: int = 1, progress_bar: Optional[bool] = True) -> Dict[str, Any]:
        """
        Generates evaluation dataset from a provided set of prompts. For each prompt,
        `self.count` responses are generated.

        Parameters
        ----------
        prompts : list of strings
            List of prompts from which LLM responses will be generated

        system_prompt : str or None, default="You are a helpful assistant."
            Optional argument for user to provide custom system prompt

        count : int, default=1
            Specifies number of responses to generate for each prompt.

        progress_bar : bool, default=True
            If True, displays a progress bar while generating responses

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
        assert all(isinstance(prompt, str) for prompt in prompts), "If using custom prompts, please ensure `prompts` is of type list[str]"
        if self.llm.temperature == 0:
            assert count == 1, "temperature must be greater than 0 if count > 1"
        self._update_count(count)
        self.system_message = SystemMessage(system_prompt)

        generations, duplicated_prompts = await self._generate_in_batches(prompts=prompts, progress_bar=progress_bar)

        responses = generations["responses"]
        logprobs = generations["logprobs"]
        return {"data": {"prompt": self._enforce_strings(duplicated_prompts), "response": self._enforce_strings(responses)}, "metadata": {"system_prompt": system_prompt, "temperature": self.llm.temperature, "count": self.count, "logprobs": logprobs}}

    def _create_tasks(self, prompts: List[str]) -> Tuple[List[Any], List[str]]:
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

    async def _generate_in_batches(self, prompts: List[str], progress_bar: Optional[bool] = True) -> Tuple[Dict[str, List[Any]], List[str]]:
        """Executes async IO with langchain in batches to avoid rate limit error"""
        assert self.count > 0, "count must be greater than 0"
        self.use_progress_bar = progress_bar
        if self.max_calls_per_min:
            batch_size = max(1, self.max_calls_per_min // self.count)
            check_batch_time = True
        else:
            batch_size = len(prompts)
        prompts_partition = list(self._split(prompts, batch_size))

        duplicated_prompts = []
        generations = {"responses": [], "logprobs": []}
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.completed}/{task.total}"), TimeElapsedColumn()) as self.progress:
            if progress_bar:
                if self.count == 1:
                    self.progress_task = self.progress.add_task(f"- {'Scoring responses with LLM-as-a-Judge' if self.is_judge else 'Generating responses'}...", total=len(prompts))
                else:
                    self.progress_task = self.progress.add_task(f"- Generating candidate responses ({self.count} per prompt)...", total=len(prompts) * self.count)
            for batch_idx, prompt_batch in enumerate(prompts_partition):
                if batch_idx == len(prompts_partition) - 1:
                    check_batch_time = False
                await self._process_batch(prompt_batch, duplicated_prompts, generations, check_batch_time)
            time.sleep(0.1)

        return generations, duplicated_prompts

    async def _process_batch(self, prompt_batch: List[str], duplicated_prompts: List[str], generations: Dict[str, List[Any]], check_batch_time: bool) -> None:
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

    async def _async_api_call(self, prompt: str, count: int = 1) -> Dict[str, Any]:
        """Generates responses asynchronously using an RunnableSequence object"""
        messages = [self.system_message, HumanMessage(prompt)]
        logprobs = [None] * count
        result = await self.llm.agenerate([messages])
        if self.use_progress_bar:
            for _ in range(count):
                self.progress.update(self.progress_task, advance=1)
        if hasattr(self.llm, "logprobs"):
            if self.llm.logprobs:
                if "logprobs_result" in result.generations[0][0].generation_info:
                    logprobs = [result.generations[0][i].generation_info["logprobs_result"] for i in range(count)]
                elif "logprobs" in result.generations[0][0].generation_info:
                    logprobs = [result.generations[0][i].generation_info["logprobs"]["content"] for i in range(count)]
        return {"logprobs": logprobs, "responses": [result.generations[0][i].text for i in range(count)]}

    @staticmethod
    def _enforce_strings(texts: List[Any]) -> List[str]:
        """Enforce that all outputs are strings"""
        return [str(r) for r in texts]

    @staticmethod
    def _split(list_a: List[str], chunk_size: int) -> Iterator[List[str]]:
        """Partitions list"""
        for i in range(0, len(list_a), chunk_size):
            yield list_a[i : i + chunk_size]
