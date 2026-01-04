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
import time
from typing import List, Optional, Callable
from uqlm.utils.prompts import get_claim_breakdown_prompt
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
import re


class ResponseDecomposer:
    def __init__(self, claim_decomposition_llm: Optional[BaseChatModel] = None, response_template: Callable = get_claim_breakdown_prompt) -> None:
        """
        Class for decomposing responses into individual claims or sentences. This class is used as an intermediate
        step for longform UQ methods.

        Parameters
        ----------
        claim_decomposition_llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        response_template: Callable
            A function that takes a response and returns a list of claims.
        """
        self.claim_decomposition_llm = claim_decomposition_llm
        self.response_template = response_template

    def decompose_sentences(self, responses: List[str], progress_bar: Optional[Progress] = None) -> List[List[str]]:
        """
        Parameters
        ----------
        responses: List[str]
            LLM response that will be decomposed into independent claims.

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses
        """
        if progress_bar:
            progress_task = progress_bar.add_task("  - Decomposing responses into sentences...", total=len(responses))

        sentence_lists = []
        for response in responses:
            if progress_bar:
                progress_bar.update(progress_task, advance=1)
            sentence_lists.append(self._get_sentences_from_response(response))
        time.sleep(0.1)
        return sentence_lists

    def decompose_candidate_sentences(self, sampled_responses: List[List[str]], progress_bar: Optional[Progress] = None) -> List[List[List[str]]]:
        """
        Parameters
        ----------
        sampled_responses: List[List[str]]
            List of lists of sampled responses to be decomposed

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses
        """
        num_responses = len(sampled_responses[0])
        if progress_bar:
            self.progress_task = progress_bar.add_task("  - Decomposing candidate responses into sentences...", total=len(sampled_responses) * num_responses)
        sampled_sentences_sets = []
        for candidates in sampled_responses:
            sentence_sets_i = self.decompose_sentences(responses=candidates)
            sampled_sentences_sets.append(sentence_sets_i)
            if progress_bar:
                progress_bar.update(self.progress_task, advance=num_responses)
        time.sleep(0.1)
        return sampled_sentences_sets

    async def decompose_claims(self, responses: List[str], response_template: Callable = None, progress_bar: Optional[Progress] = None) -> List[List[str]]:
        """
        Parameters
        ----------
        responses: List[str]
            LLM response that will be decomposed into independent claims.

        response_template: Callable
            A function that takes a response and returns a list of claims.

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses
        """
        if response_template is not None:
            self.response_template = response_template
        if not self.claim_decomposition_llm:
            raise ValueError("llm must be provided to decompose responses into claims")
        if progress_bar:
            self.progress_task = progress_bar.add_task("  - Decomposing responses into claims...", total=len(responses))
        claim_sets = await self._decompose_claims(responses=responses, progress_bar=progress_bar)
        time.sleep(0.1)
        return claim_sets

    async def decompose_candidate_claims(self, sampled_responses: List[List[str]], progress_bar: Optional[Progress] = None) -> List[List[List[str]]]:
        """
        Parameters
        ----------
        sampled_responses: List[List[str]]
            List of lists of sampled responses to be decomposed

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses
        """
        if not self.claim_decomposition_llm:
            raise ValueError("llm must be provided to decompose candidate responses into claims")
        num_responses = len(sampled_responses[0])
        if progress_bar:
            self.progress_task = progress_bar.add_task("  - Decomposing candidate responses into claims...", total=len(sampled_responses) * num_responses)
        tasks = [self._decompose_claims(responses=candidates, progress_bar=progress_bar, matched_claims=True) for candidates in sampled_responses]
        sampled_claim_sets = await asyncio.gather(*tasks)
        time.sleep(0.1)
        return sampled_claim_sets

    async def _decompose_claims(self, responses: List[str], progress_bar: Optional[Progress] = None, matched_claims: bool = True) -> List[str]:
        """Helper for decomposing list of responses into claims"""
        if not matched_claims:
            progress_bar.update(self.progress_task, advance=1)
            progress_bar_use = None
        else:
            progress_bar_use = progress_bar
        tasks = [self._get_claims_from_response(response=response, progress_bar=progress_bar_use) for response in responses]
        return await asyncio.gather(*tasks)

    def _get_sentences_from_response(self, text: str) -> list[str]:
        """
        A more sophisticated approach inspired by NLTK's sentence tokenizer.
        Uses multiple passes and heuristics.
        """
        text = re.sub(r"(\d+)\.(\d+)", r"\1<DECIMAL>\2", text)
        abbrev_pattern = r"\b(?:mr|mrs|ms|dr|prof|sr|jr|vs|etc|inc|ltd|corp|co|st|ave|blvd|rd|ph\.d|m\.d|b\.a|m\.a|u\.s|u\.k|n\.y|l\.a|d\.c)\."

        abbreviations = re.finditer(abbrev_pattern, text, re.IGNORECASE)
        protected_text = text
        offset = 0

        for match in abbreviations:
            start, end = match.span()
            start += offset
            end += offset
            replacement = match.group().replace(".", "<DOT>")
            protected_text = protected_text[:start] + replacement + protected_text[end:]
            offset += len(replacement) - len(match.group())

        pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(pattern, protected_text.strip())

        for i, sentence in enumerate(sentences):
            sentence = sentence.replace("<DOT>", ".")
            sentence = sentence.replace("<DECIMAL>", ".")
            sentences[i] = sentence.strip()
        return sentences

    async def _get_claims_from_response(self, response: str, progress_bar: Optional[Progress] = None) -> List[str]:
        """Decompose single response into claims using LLM and extract claims from the result"""
        decomposed_response = await self.claim_decomposition_llm.ainvoke(self.response_template(response))
        if progress_bar:
            progress_bar.update(self.progress_task, advance=1)

        llm_response = decomposed_response.content
        if self._is_none_response(llm_response):
            return []

        claim_pattern = r"(?:^|\n)\s*###\s*(.+?)(?=\n\s*###|\n\s*$|$)"
        matches = re.findall(claim_pattern, llm_response, re.MULTILINE | re.DOTALL)

        claims = []
        for match in matches:
            cleaned_claim = re.sub(r"\s+", " ", match.strip())
            if cleaned_claim:  # Skip empty claims
                claims.append(cleaned_claim)

        return claims

    def _is_none_response(self, llm_response: str) -> bool:
        """
        Check if the LLM response indicates no claims are present. Detects the template-instructed "### NONE" response and common variations.
        """
        return bool(re.search(r"###\s*none\b", llm_response.strip(), re.IGNORECASE))
