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

import json
from typing import Callable, List, Optional, Any, Union
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.utils.results import UQResult
from uqlm.scorers import BlackBoxUQ
from uqlm.scorers.longform.baseclass.uncertainty import LongFormUQ


class LongTextCF(LongFormUQ):
    def __init__(
        self,
        llm: BaseChatModel,
        scorers: Optional[List[str]] = None,
        granularity: str = "claim",
        system_prompt: str = "You are a helpful assistant.",
        claim_decomposition_llm: BaseChatModel = None,
        claim_decomposition_prompt: Union[str, Callable] = "zhang_2025",
        frequency_scorer_llm: BaseChatModel = None,
        n_frequency_responses: int = 5,
        sampling_temperature: float = 1.0,
        max_calls_per_min: Optional[int] = None,
        frequency_scorer_max_calls_per_min: Optional[int] = None,
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

        claim_decomposition_prompt : Union[str, Callable], default="zhang_2025"
            Specifies the prompt template used to decompose responses into atomic claims. Accepts one of the
            following string keys: ``"zhang_2025"``, ``"farquhar_2024"``, ``"mohri_2024"``, ``"jiang_2024"``,
            or a custom callable with signature ``(response: str) -> str``. Only applies when
            ``granularity="claim"``.

        frequency_scorer_llm : langchain `BaseChatModel`, default=None
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
        super().__init__(
            llm=llm, granularity=granularity, scorers=self.scorers, response_refinement=True, claim_decomposition_llm=claim_decomposition_llm, claim_decomposition_prompt=claim_decomposition_prompt, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param
        )
        self.bb_object = BlackBoxUQ(llm=llm, scorers=self.scorers, device=device, max_calls_per_min=max_calls_per_min, max_length=max_length)
        self.reconstructor_result = {}
        self.sampling_temperature = sampling_temperature
        self.frequency_scorer_llm = frequency_scorer_llm
        self.frequency_scorer_max_calls_per_min = frequency_scorer_max_calls_per_min
        self.n_alternate_responses = n_frequency_responses
        # print('llm temperature:', self.llm.temperature)
        """TODO:
            - figure out temps
            - figure out when to use bb_object or parent class
            
            - add gpt scoring
            - optimize rather than looping is possible
            - clean up unused params, docstrings, typing, comments
            - add progress bars every including frequency scoring
            - add new prompt templates
        """

    async def generate_and_score(self, prompts: List[str], response_refinement_threshold: float = 1 / 3, claim_scorers: List[str] = ["frequency"], show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Generate and score the responses.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        response_refinement_threshold : float, default=1/3
            Threshold for uncertainty-aware filtering. Claims with confidence scores below this threshold are dropped from the
            refined response. Only used if response_refinement is True.

        show_progress_bars : bool, default=True
            If True, displays progress bars while generating and scoring responses.
        """
        self._construct_progress_bar(show_progress_bars)
        self._display_generation_header(show_progress_bars)

        responses = await self.generate_original_responses(prompts=prompts, progress_bar=self.progress_bar)
        scores = await self.score(prompts=prompts, responses=responses, response_refinement_threshold=response_refinement_threshold, claim_scorers=claim_scorers, show_progress_bars=show_progress_bars)
        return scores

    async def score(self, prompts: List[str], responses: List[str], response_refinement_threshold: float = 1 / 3, claim_scorers: List[str] = ["frequency"], show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Decompose responses, generate questions for each claim/sentence, sample LLM responses to the questions, and score consistency on those generated answers to measure confidence.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        responses : list of str
            A list of model responses for the prompts.

        response_refinement_threshold : float, default=1/3
            Threshold for uncertainty-aware filtering. Claims with confidence scores below this threshold are dropped from the
            refined response. Only used if response_refinement is True.

        show_progress_bars : bool, default=True
            If True, displays a progress bar while scoring responses
        """
        self.prompts = prompts
        self.responses = responses

        self._construct_progress_bar(show_progress_bars)

        # 1 decompose - use existing
        await self._decompose_responses(show_progress_bars)
        # print(self.claim_sets)

        # 2 score subclaims - populates claims_data
        self.claim_scores = await self._score_from_decomposed(
            prompts=self.prompts,
            claim_sets=self.claim_sets,
            claim_scorers=claim_scorers,
            progress_bar=self.progress_bar
        )
        # print("claim_scores:")
        # print(self.claim_scores)

        # 3 select above threshold &
        # 4 reconstruct - use existing
        # TODO: allow user to specify another llm for reconstruction, or always use the decomposer? 
        self.reconstructor_result = await self.reconstructor.reconstruct_responses(
            claim_sets=self.claim_sets,
            claim_scores=self.claim_scores["aggregated"],
            threshold=response_refinement_threshold,
            progress_bar=self.progress_bar
        )
        # print("Reconstructed responses:")   
        # print(self.reconstructor_result)
        refined_responses = self.reconstructor_result["refined_responses"]
        refined_responses = [
            resp if resp is not None else "" for resp in refined_responses
        ]
        self.reconstructor_result["refined_responses"] = refined_responses

        # # 5 score against original response
        bb_result = self.bb_object.score(
            responses,
            [[rr] for rr in refined_responses],
            show_progress_bars=None,
            _display_header=False
        )
        # print("BB result:")
        # print(bb_result.to_dict())

        self.scores_dict = self._process_bb_result(bb_result=bb_result)
        # print("Scores dict:")
        # print(self.scores_dict)

        self._stop_progress_bar()
        self.progress_bar = None

        return self._construct_result()

    async def _score_from_decomposed(self, prompts: List[str],  claim_sets: List[List[str]], claim_scorers: List[str], aggregation: str = "sum", progress_bar: Optional[Progress] = None) -> dict:
        scores = {}
        if "frequency" in claim_scorers:
            scores["frequency"] = await self._get_frequency_scores(claim_sets, prompts)
        if "gpt" in claim_scorers:
            scores["gpt"] = await self._get_gpt_scores(claim_sets, prompts)
        # make a list of summed scores for each claim set
        scores["aggregated"] = [[0.0] * len(claim_sets[i]) for i in range(len(prompts))]
        if aggregation == "sum":
            for i in range(len(prompts)):
                for j in range(len(claim_sets[i])):
                    scores["aggregated"][i][j] = sum(scores[claim_scorer][i][j] for claim_scorer in claim_scorers)
        elif aggregation == "mean":
            for i in range(len(prompts)):
                for j in range(len(claim_sets[i])):
                    scores["aggregated"][i][j] = sum(scores[claim_scorer][i][j] for claim_scorer in claim_scorers) / len(claim_scorers)
        return scores

    async def _get_frequency_scores(
        self, claim_sets: List[List[str]], prompts: List[str], progress_bar: Optional[Progress] = None
    ) -> List[List[float]]:
        
        #  Generate n_samples alternate outputs with temperature 1.0.
        # TODO: allow user to specify another llm here?
        self.alternate_responses = await self.generate_candidate_responses(
            prompts=prompts, num_responses=self.n_alternate_responses, progress_bar=progress_bar
        )
        # print("Alternate responses:")
        # print(self.alternate_responses)

        frequency_scores = [[0.0] * len(claim_sets[i]) for i in range(len(prompts))]

        llm = self.llm
        self.max_calls_per_min = self.max_calls_per_min
        if self.frequency_scorer_llm:
            self.llm = self.frequency_scorer_llm
            if self.frequency_scorer_max_calls_per_min:
                self.max_calls_per_min = self.frequency_scorer_max_calls_per_min

        for i in range(len(prompts)):
            claim_string = "\n".join(
                [str(j) + ": " + fact for j, fact in enumerate(claim_sets[i])]
            )

            # Count the number of times the alternate outputs support the sub-claims (using LM).
            for alternate_response in self.alternate_responses[i]:
                counting_prompt = (
                    'You will get a list of claims and piece of text. For each claim, score whether the text supports, contradicts, or is unrelated to the claim. Directly return a jsonl, where each line is {"id":CLAIM_ID, "score":SCORE}. Directly return the jsonl with no explanation or other formatting. For the SCORE, return 1 for supports, -1 for contradicts, and 0 for unrelated. The claims are:\n'
                    + claim_string
                    + "\n\nThe text is:\n"
                    + alternate_response
                )  # removed the brackets around CLAIM_ID and SCORE to match the expected output format
                # output = query_model(
                #     client, counting_prompt, model, max_tokens=1000, temperature=0
                # )
                output = await self._generate_responses(prompts=[counting_prompt], count=1, temperature=0, progress_bar=progress_bar)
                # print("response output:")
                # print(output)
                output = output['responses'][0]
                output = output.replace("```jsonl\n", "")
                output = output.replace("```", "")
                # print("response output after cleanup:")
                # print(output)
                try:
                    for line in output.splitlines():
                        scores = json.loads(line)
                        idx = int(scores["id"])
                        frequency_scores[i][idx] += float(scores["score"])
                except Exception as ex:
                    print(ex)
                    print("Failed to parse as jsonl")
                    print(output)

        self.llm = llm  # restore original llm
        self.max_calls_per_min = self.max_calls_per_min  # restore original max_calls_per_min

        return frequency_scores

    def _process_bb_result(self, bb_result: UQResult) -> dict:
        """Process the result from the black box scorer and return a dictionary of scores."""
        bb_result_dict = bb_result.to_dict()
        data = bb_result_dict.get("data", {})
        scores_dict = {}
        for key in self.bb_object.scorers:
            if key in data:
                scores_dict[key] = data[key]
        return scores_dict

    def _extract_claim_data(self) -> None:
        """Extract claims data"""
        claims_data = []
        for i in range(len(self.claim_sets)):
            claim_i_data = []
            for j in range(len(self.claim_sets[i])):
                claims_dict = {
                    self.granularity: self.claim_sets[i][j],
                    "removed": (
                        False
                        if not self.reconstructor_result
                        else self.reconstructor_result["removed"][i][j]
                    ),
                    "aggregated_score": self.claim_scores["aggregated"][i][j]
                }
                if "frequency" in self.claim_scores:
                    claims_dict["frequency_score"] = self.claim_scores["frequency"][i][j]
                if "gpt" in self.claim_scores:
                    claims_dict["gpt_score"] = self.claim_scores["gpt"][i][j]
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
        data["claims_data"] = self._extract_claim_data()
        if self.alternate_responses:
            data["frequency_responses_lists"] = self.alternate_responses
        if self.reconstructor_result:
            data["refined_responses"] = self.reconstructor_result.get("refined_responses")
        data.update(self.scores_dict)
        result = {"data": data, "metadata": {"granularity": self.granularity, "temperature": None if not self.llm else self.llm.temperature, "sampling_temperature": self.bb_object.sampling_temperature, "num_alternate_responses": self.n_alternate_responses, "response_refinement_threshold": self.response_refinement_threshold}}
        return UQResult(result)
