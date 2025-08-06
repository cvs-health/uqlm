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
from uqlm.utils.prompt_templates import get_entailment_template
from uqlm.utils.postprocessors import claims_postprocessor
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier

class FactScoreBenchmark:
    def __init__(self, judge_llm: BaseChatModel):
        self.judge_llm = judge_llm

    def evaluate_scorers(self, 
                        scorers: list[UncertaintyQuantifier] | UncertaintyQuantifier, 

                        sampling_temperature: float = 0.7, 
                        num_responses: int = 5,
                        progress: bool = True) -> dict:
        """
        Parameters
        ----------
        scorers: list[UncertaintyQuantifier] | UncertaintyQuantifier
            List of scorers to be evaluated.
        sampling_temperature: float
            The temperature to use for sampling responses.
        num_responses: int
            The number of responses to sample.
        progress: bool
            Whether to show a progress bar.
        Returns
        -------
        dict
            Dictionary containing the results of the evaluation.
        """
        results = {}
        if isinstance(llms, BaseChatModel):
            llms = [llms]
        if isinstance(scorers, UncertaintyQuantifier):
            scorers = [scorers]
        for llm in llms:
            for scorer in scorers:
                # get greedy response

                # get sampled responses


                scorer.llm = llm
                scorer.temperature = temperature
                scorer.num_responses = num_responses
                scorer.progress = progress
                scorer.evaluate()



        return scorer.results

    def find_entailment(self, claims: list[str] | str, source_texts: list[str] | str) -> list[dict]:
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
        entailments = []
        for claim in claims:
            for source_text in source_texts:
                prompt = get_entailment_template(claim, source_text)
                entailments.append(self.judge_llm.invoke(prompt))
        return entailments
