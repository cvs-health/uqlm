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

from typing import List, Optional
from rich.progress import Progress
from uqlm.utils.prompts.claims_prompts import get_claim_dedup_prompt
from uqlm.utils.response_generator import ResponseGenerator
from langchain_core.language_models.chat_models import BaseChatModel
import re


class ClaimMerger:
    def __init__(self, claim_merging_llm: BaseChatModel) -> None:
        self.rg = ResponseGenerator(llm=claim_merging_llm)

    async def merge_claims(self, original_claim_sets: List[List[str]], sampled_claim_sets: List[List[List[str]]], progress_bar: Optional[Progress] = None) -> List[List[str]]:
        """Process claim deduplication for response sets.

        Leverages ResponseGenerator's ability to handle multiple prompts at once
        by collecting dedup prompts and making batch calls.

        If sampled_claim_sets contains only empty lists and entailment_score_sets is provided,
        infers master claims from entailment_score_sets keys. Otherwise returns original_claim_sets.
        """
        num_response_sets = len(original_claim_sets)
        num_samples = len(sampled_claim_sets[0]) * num_response_sets

        self.progress_task = None
        if progress_bar:
            self.progress_task = progress_bar.add_task("  - Deduplicating claims across candidate responses...", total=num_samples)

        self.master_claim_sets = [original_claim_sets[i] for i in range(num_response_sets)]
        for iteration in range(num_samples):
            prompts, prompt_metadata = self._construct_merging_prompts(sampled_claim_sets=sampled_claim_sets, iteration=iteration)

            # Batch call for this iteration across all response sets
            if prompts:
                result = await self.rg.generate_responses(prompts=prompts)
                responses = result["data"]["response"]
            else:
                responses = []

            self._process_claim_merging_generations(responses, prompt_metadata, progress_bar)

        return self.master_claim_sets

    def _process_claim_merging_generations(self, responses, prompt_metadata, progress_bar) -> None:
        # Update master_claim_sets with results
        response_idx = 0
        for i, has_prompt, current_master, sampled_claims in prompt_metadata:
            if has_prompt and response_idx < len(responses):
                response_text = responses[response_idx]
                new_claims = re.findall(r"^\s*-\s*(.+)", response_text, re.MULTILINE)

                if new_claims:
                    self.master_claim_sets[i] = current_master + new_claims

                response_idx += 1

        if progress_bar and self.progress_task is not None:
            progress_bar.update(self.progress_task, advance=len(responses))

    def _construct_merging_prompts(self, sampled_claim_sets, iteration):
        prompts = []
        prompt_metadata = []  # (response_set_idx, has_prompt)

        for i in range(len(self.master_claim_sets)):
            if iteration < len(sampled_claim_sets[i]):
                sampled_claims = sampled_claim_sets[i][iteration]
                unique_sampled_claims = list(set(sampled_claims) - set(self.master_claim_sets[i]))

                if unique_sampled_claims:
                    prompts.append(get_claim_dedup_prompt(self.master_claim_sets[i], unique_sampled_claims))
                    prompt_metadata.append((i, True, self.master_claim_sets[i], sampled_claims))
                else:
                    prompt_metadata.append((i, False, self.master_claim_sets[i], sampled_claims))
            else:
                prompt_metadata.append((i, False, self.master_claim_sets[i], []))

        return prompts, prompt_metadata
