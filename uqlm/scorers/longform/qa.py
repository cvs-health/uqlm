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

import numpy as np
from typing import List, Optional, Any
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.longform.qa.question_generator import QuestionGenerator
from uqlm.utils.prompts.claim_qa import get_answer_template
from uqlm.utils.results import UQResult
from uqlm.scorers import BlackBoxUQ
from uqlm.scorers.longform.baseclass.uncertainty import LongFormUQ


class LongTextQA(LongFormUQ):
    def __init__(
        self,
        llm: BaseChatModel,
        scorers: Optional[List[str]] = None,
        granularity: str = "claim",
        aggregation: str = "mean",
        response_refinement: bool = False,
        claim_filtering_scorer: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        claim_decomposition_llm: BaseChatModel = None,
        question_generator_llm: BaseChatModel = None,
        sampling_temperature: float = 1.0,
        max_calls_per_min: Optional[int] = None,
        questioner_max_calls_per_min: Optional[int] = None,
        max_length: int = 1000,
        device: Any = None,
        use_n_param: bool = False,
    ):
        """
        Implements a generalization of the longform semantic entropy approach by Farquhar et al. (2024): https://www.nature.com/articles/s41586-024-07421-0.

        Parameters
        ----------
        llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        scorers : subset of {"entailment", "noncontradiction", "contrasted_entailment", "bert_score", "cosine_sim"}, default=None
            Specifies which black box (consistency) scorers to include. If None, defaults to ["entailment"].

        granularity : str, default="claim"
            Specifies whether to decompose and score at claim or sentence level granularity. Must be either "claim" or "sentence"

        aggregation : str, default="mean"
            Specifies how to aggregate claim/sentence-level scores to response-level scores. Must be one of 'min' or 'mean'.

        response_refinement : bool, default=False
            Specifies whether to refine responses with uncertainty-aware decoding. This approach removes claims with confidence
            scores below the response_refinement_threshold and uses the claim_decomposition_llm to reconstruct the response from
            the retained claims. Only available for claim-level granularity. For more details, refer to
            Jiang et al., 2024: https://arxiv.org/abs/2410.20783

        claim_filtering_scorer : Optional[str], default=None
            specifies which scorer to use to filter claims if response_refinement is True. If not provided, defaults to the first
            element of self.scorers.

        claim_decomposition_llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel` to be used for decomposing responses into individual claims. Also used for claim refinement.
            If granularity="claim" and claim_decomposition_llm is None, the provided `llm` will be used for claim decomposition.

        question_generator_llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel` to be used for decomposing responses into individual claims. Used for generating questions
            from claims or sentences in claim-QA approach. If None, defaults to claim_decomposition_llm.

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. Applies to 'luq', 'luq_atomic'
            scorers. Pass a torch.device to leverage GPU.

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        system_prompt : str or None, default="You are a helpful assistant."
            Optional argument for user to provide custom system prompt

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.

        sampling_temperature : float, default=1.0
            The 'temperature' parameter for llm model to generate sampled LLM responses. Must be greater than 0.

        use_n_param : bool, default=False
            Specifies whether to use `n` parameter for `BaseChatModel`. Not compatible with all
            `BaseChatModel` classes. If used, it speeds up the generation process substantially when num_responses > 1.

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError
        """
        self.scorers = ["semantic_negentropy"] if not scorers else scorers
        super().__init__(llm=llm, granularity=granularity, aggregation=aggregation, scorers=self.scorers, response_refinement=response_refinement, claim_filtering_scorer=claim_filtering_scorer, claim_decomposition_llm=claim_decomposition_llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param)
        self.question_generator = QuestionGenerator(question_generator_llm=question_generator_llm if question_generator_llm is not None else self.decomposer.claim_decomposition_llm, max_calls_per_min=questioner_max_calls_per_min)
        self.bb_object = BlackBoxUQ(llm=llm, scorers=self.scorers, device=device, max_calls_per_min=max_calls_per_min, sampling_temperature=sampling_temperature, max_length=max_length)
        self.uad_result = {}

    async def generate_and_score(self, prompts: List[str], num_questions: int = 1, num_claim_qa_responses: int = 5, response_refinement_threshold: float = 1 / 3, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Generate and score the responses.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        num_questions : int, default=1
            The number of questions to generate for each claim/sentence.

        num_claim_qa_responses : int, default=5
            The number of responses to generate for each claim-inverted question.

        response_refinement_threshold : float, default=1/3
            Threshold for uncertainty-aware filtering. Claims with confidence scores below this threshold are dropped from the
            refined response. Only used if response_refinement is True.

        show_progress_bars : bool, default=True
            If True, displays progress bars while generating and scoring responses.
        """
        self._construct_progress_bar(show_progress_bars)
        self._display_generation_header(show_progress_bars)

        responses = await self.generate_original_responses(prompts=prompts, progress_bar=self.progress_bar)
        return await self.score(prompts=prompts, responses=responses, num_questions=num_questions, num_claim_qa_responses=num_claim_qa_responses, response_refinement_threshold=response_refinement_threshold, show_progress_bars=show_progress_bars)

    async def score(self, prompts: List[str], responses: List[str], num_questions: int = 1, num_claim_qa_responses: int = 5, response_refinement_threshold: float = 1 / 3, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Decompose responses, generate questions for each claim/sentence, sample LLM responses to the questions, and score consistency on those generated answers to measure confidence.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        responses : list of str
            A list of model responses for the prompts.

        num_questions : int, default=1
            The number of questions to generate for each claim/sentence.

        num_claim_qa_responses : int, default=5
            The number of responses to generate for each claim-inverted question.

        response_refinement_threshold : float, default=1/3
            Threshold for uncertainty-aware filtering. Claims with confidence scores below this threshold are dropped from the
            refined response. Only used if response_refinement is True.

        show_progress_bars : bool, default=True
            If True, displays a progress bar while scoring responses
        """
        self.prompts = prompts
        self.responses = responses

        self._construct_progress_bar(show_progress_bars)
        await self._decompose_responses(show_progress_bars)

        result = await self._score_from_decomposed(prompts=self.prompts, num_questions=num_questions, num_claim_qa_responses=num_claim_qa_responses, claim_sets=self.claim_sets, response_refinement_threshold=response_refinement_threshold, progress_bar=self.progress_bar)
        return result

    async def _score_from_decomposed(self, claim_sets: List[List[str]], prompts: Optional[List[str]] = None, num_questions: int = 1, num_claim_qa_responses: int = 5, response_refinement_threshold=1 / 3, progress_bar: Optional[Progress] = None):
        """
        Evaluate the ClaimQA scores for a given set of prompts and claim_sets.
        """
        self.num_questions = num_questions
        self.num_claim_qa_responses = num_claim_qa_responses
        self.claim_sets = claim_sets
        self.prompts = [None] * len(claim_sets) if not prompts else prompts

        prompts_flat = []
        for prompt, claim_set in zip(self.prompts, self.claim_sets):
            prompts_flat.extend([prompt] * len(claim_set) * self.num_questions)
        num_claims = [len(claim_set) for claim_set in self.claim_sets]

        generated_questions = await self.question_generator.generate_questions(claim_sets=self.claim_sets, num_questions=self.num_questions, progress_bar=progress_bar)
        formatted_claim_questions = [get_answer_template(claim_question=generated_questions[i], original_question=prompts_flat[i]) for i in range(len(generated_questions))]

        self.bb_object.progress_bar = progress_bar
        self.bb_object.generation_type = "claim_qa"
        bb_result = await self.bb_object.generate_and_score(prompts=formatted_claim_questions, num_responses=self.num_claim_qa_responses, show_progress_bars=True if progress_bar else False)
        self.scores_dict = self._process_bb_result(bb_result=bb_result, formatted_claim_questions=generated_questions, num_claims=num_claims)

        if self.response_refinement:
            self.uad_result = await self.uncertainty_aware_decode(claim_sets=self.claim_sets, claim_scores=self.claim_scores[self.uad_scorer], response_refinement_threshold=response_refinement_threshold, show_progress_bars=True if progress_bar else False)

        self.scores_dict["claims_data"] = self._extract_claim_data()

        if "removed" in self.uad_result:
            del self.uad_result["removed"]

        self._stop_progress_bar()
        self.progress_bar = None

        return self._construct_result()

    def _process_bb_result(self, bb_result: Any, formatted_claim_questions: List[str], num_claims: List[float]) -> None:
        """Format BlackBoxUQ output data"""
        self.claim_scores = {key: [] for key in self.bb_object.scorers}
        self.response_fact_questions, self.response_fact_questions_responses, self.response_fact_questions_sampled_responses = [], [], []

        initial_index = 0
        for i in range(len(self.claim_sets)):
            self.response_fact_questions.append([formatted_claim_questions[j : j + self.num_questions] for j in range(initial_index, initial_index + num_claims[i] * self.num_questions, self.num_questions)])
            tmp_data = bb_result.to_dict()["data"]
            self.response_fact_questions_responses.append([tmp_data["responses"][j : j + self.num_questions] for j in range(initial_index, initial_index + num_claims[i] * self.num_questions, self.num_questions)])
            self.response_fact_questions_sampled_responses.append([tmp_data["sampled_responses"][j : j + self.num_questions] for j in range(initial_index, initial_index + num_claims[i] * self.num_questions, self.num_questions)])
            for key in self.bb_object.scorers:
                tmp = bb_result.to_dict()["data"][key][initial_index : initial_index + num_claims[i] * self.num_questions]
                if self.num_questions == 1:
                    tmp_claim_scores = tmp
                else:
                    tmp_claim_scores = [np.mean(tmp[j * self.num_questions : (j + 1) * self.num_questions]) for j in range(num_claims[i])]
                self.claim_scores[key].append(tmp_claim_scores)
            initial_index += num_claims[i] * self.num_questions
        scores_dict = {key: self._aggregate_scores(scores) for key, scores in self.claim_scores.items()}
        return scores_dict

    def _extract_claim_data(self) -> None:
        """Extract claims data"""
        claims_data = []
        for i in range(len(self.claim_sets)):
            claim_i_data = []
            for j in range(len(self.claim_sets[i])):
                claims_dict = {self.granularity: self.claim_sets[i][j], "removed": False if not self.uad_result else self.uad_result["removed"][i][j], "claim_questions": self.response_fact_questions[i][j], "claim_qa_responses": self.response_fact_questions_responses[i][j], "claim_qa_sampled_responses": self.response_fact_questions_sampled_responses[i][j]}
                for scorer in self.bb_object.scorers:
                    claims_dict[scorer] = self.claim_scores[scorer][i][j]
                claim_i_data.append(claims_dict)
            claims_data.append(claim_i_data)
        return claims_data

    def _construct_result(self) -> Any:
        """Constructs UQResult object"""
        data = {}
        if self.prompts:
            data["prompts"] = self.prompts
        if self.responses:
            data["responses"] = self.responses

        data.update(self.scores_dict)
        data.update(self.uad_result)
        result = {"data": data, "metadata": {"granularity": self.granularity, "aggregation": self.aggregation, "temperature": None if not self.llm else self.llm.temperature, "sampling_temperature": self.bb_object.sampling_temperature, "num_claim_qa_responses": self.num_claim_qa_responses, "response_refinement_threshold": self.response_refinement_threshold}}
        return UQResult(result)
