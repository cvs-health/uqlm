import asyncio
import contextlib
import io
import re
import numpy as np
from typing import List, Optional, Any
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.utils.response_generator import ResponseGenerator
from uqlm.longform.decomposition.response_decomposer import ResponseDecomposer
from uqlm.utils.prompt_templates import get_factoid_template, get_question_template, get_answer_template, get_claim_breakdown_template
from uqlm.utils.results import UQResult
from uqlm.scorers import BlackBoxUQ


class ClaimQAScorer:
    def __init__(
        self,
        llm: BaseChatModel,
        llm_decomposer: BaseChatModel = None,
        llm_questioner: BaseChatModel = None,
        black_box_scorers: Optional[List[str]] = None,
        response_template: str = "atomic",
        device: Any = None,
        system_prompt: str = "You are a helpful assistant.",
        sampling_temperature: float = 1.0,
        max_calls_per_min: int = 1000,
        use_n_param: bool = False,
        num_questions: int = 1,
        num_claim_qa_responses: int = 5,
    ):
        """
        Initialize the ClaimQAScorer.

        Parameters
        ----------
        llm : BaseChatModel
            The original LLM to use for generating responses.
        llm_decomposer : BaseChatModel
            The LLM to use for decomposing the claims.
        llm_questioner : BaseChatModel
            The LLM to use for generating questions.
        response_template : str, default="atomic"
            The template to use for generating responses. Choose from "atomic" or "factoid".
        system_prompt : str
            The system prompt to use for generating responses. Default is "You are a helpful assistant."
        sampling_temperature : float, default=1.0
            The temperature to use for sampling the responses.
        max_calls_per_min : int
            The maximum number of calls per minute to the LLM.
        use_n_param : bool
            Whether to use the n parameter for the LLM.
        num_questions : int, default=2
            The number of questions to generate for each factoid.
        num_claim_qa_responses : int, default=2
            The number of responses to generate for each claim-inverted question.
        """
        self.llm = llm
        self.llm_decomposer = llm_decomposer if llm_decomposer is not None else llm
        self.llm_questioner = llm_questioner if llm_questioner is not None else self.llm_decomposer
        self.bb_object = BlackBoxUQ(llm=llm, scorers=black_box_scorers, device=device, max_calls_per_min=max_calls_per_min, sampling_temperature=sampling_temperature)
        self.system_prompt = system_prompt
        self.max_calls_per_min = max_calls_per_min
        self.use_n_param = use_n_param
        self.num_questions = num_questions # TODO: Currently, class is designed for num_questions = 1
        self.num_claim_qa_responses = num_claim_qa_responses
        if response_template == "atomic":
            self.response_template = get_claim_breakdown_template
        elif response_template == "factoid":
            self.response_template = get_factoid_template
        else:
            raise ValueError("""response_template must be either "atomic" or "factoid".""")

    async def evaluate(self, prompts: List[str], responses: List[str], factoids: List[List[str]]):
        """
        Evaluate the ClaimQA scores for a given set of prompts, responses, and factoids.
        """
        # TODO: Add progress bar
        self.prompts = prompts
        self.responses = responses
        self.factoids = factoids

        self.response_scores, self.factoid_scores = {key: [] for key in self.bb_object.scorers}, {key: [] for key in self.bb_object.scorers}
        self.response_fact_questions, self.response_fact_questions_responses, self.response_fact_questions_sampled_responses = [], [], []
        
        # Count number of claims/factoids per response
        num_factoids = [len(factoids_i) for factoids_i in self.factoids]
        print("Number of factoids per response: ", num_factoids)

        # Generate question per factoid
        prompt_to_generate_questions = [get_question_template(factoid_i) for j in range(len(self.responses)) for factoid_i in self.factoids[j]]
        generated_questions = await self._generate_responses(llm=self.llm_questioner, prompts=prompt_to_generate_questions)

        factoid_questions = [get_answer_template(claim_question=generated_question) for generated_question in generated_questions["responses"]]
        print("Number of total questions: ", len(factoid_questions))

        # Generate responses for all questions from all factoids obtained from all responses
        bb_result = await self.bb_object.generate_and_score(prompts=factoid_questions, num_responses=self.num_claim_qa_responses, show_progress_bars=False)
        print("BB result: ", bb_result.to_dict()["data"]["exact_match"])
        print("Length of BB result: ", len(bb_result.to_dict()["data"]["exact_match"]))

        initial_index = 0
        for i in range(len(self.responses)):
            self.response_fact_questions.append(bb_result.to_dict()["data"]["responses"][initial_index:initial_index + num_factoids[i]])
            self.response_fact_questions_responses.append(bb_result.to_dict()["data"]["sampled_responses"][initial_index:initial_index + num_factoids[i]])
            for key in self.bb_object.scorers:
                tmp = bb_result.to_dict()["data"][key][initial_index:initial_index + num_factoids[i]]
                self.response_scores[key].append(np.mean(tmp))
                self.factoid_scores[key].append(tmp)
            initial_index += num_factoids[i]

        return self._construct_result()

    async def generate_and_score(self, prompts: List[str], progress_bar: Optional[Progress] = None):
        """
        Generate and score the responses.

        Parameters
        ----------
        prompts : List[str]
            A list of prompts to generate responses from LLM.
        progress_bar : Optional[Progress], default=None
            A progress bar to display the progress of the generation.
        """
        self.prompts = prompts
        responses = await self._generate_responses(llm=self.llm, prompts=self.prompts, count=1, progress_bar=progress_bar)
        self.responses = responses["responses"]
        return await self.score(prompts=self.prompts, responses=self.responses, progress_bar=progress_bar)

    async def score(self, prompts: List[str], responses: List[str], progress_bar: Optional[Progress] = None):
        """
        Evaluate the QuesAns scores for a given set of factoids.

        Parameters
        ----------
        responses : List[str]
            A list of responses to be scored.
        progress_bar : Optional[Progress], default=None
            A progress bar to display the progress of the evaluation.
        """
        # Store responses if not already set
        self.prompts = prompts
        self.responses = responses
        # if progress_bar:
        #     progress_task = progress_bar.add_task("  - Decomposing responses into factoids...", total=len(responses))

        decomposer = ResponseDecomposer(claim_decomposition_llm=self.llm_decomposer, response_template=self.response_template)
        
        tasks = [decomposer.decompose_claims(responses=[response], progress_bar=progress_bar) for response in responses]
        tmp = await asyncio.gather(*tasks)
        self.factoids = [t[0] for t in tmp]

        return await self.evaluate(prompts=self.prompts, responses=responses, factoids=self.factoids)

    async def _generate_responses(self, llm, prompts: List[str], count: int = 1, progress_bar: Optional[Progress] = None) -> List[str]:
        """Helper function to generate responses with LLM.

        Parameters
        ----------
        llm : BaseChatModel
            The LLM to use for generating responses.
        prompts : List[str]
            A list of prompts to generate responses from LLM.
        count : int
            The number of responses to generate.
        progress_bar : Optional[Progress], default=None
            A progress bar to display the progress of the generation.

        Returns
        -------
        List[str]
            A list of responses generated by the LLM.
        """
        try:
            generator_object = ResponseGenerator(llm=llm, max_calls_per_min=self.max_calls_per_min, use_n_param=self.use_n_param)
            with contextlib.redirect_stdout(io.StringIO()):
                generations = await generator_object.generate_responses(prompts=prompts, count=count, system_prompt=self.system_prompt, progress_bar=progress_bar)
        except Exception:
            if progress_bar:
                progress_bar.stop()
            raise
        return {"responses": generations["data"]["response"], "logprobs": generations["metadata"]["logprobs"]}

    def _construct_result(self) -> Any:
        """Constructs UQResult object"""
        prompts = getattr(self, "prompts", [])
        responses = getattr(self, "responses", [])
        response_scores = getattr(self, "response_scores", [])
        response_fact_questions = getattr(self, "response_fact_questions", [])
        response_fact_questions_responses = getattr(self, "response_fact_questions_responses", [])
        response_fact_questions_sampled_responses = getattr(self, "response_fact_questions_sampled_responses", [])
        factoid_scores = getattr(self, "factoid_scores", [])
        data = {"prompts": prompts, "responses": responses}
        tmp = {}
        for key in response_scores:
            tmp["response_scores_"+key] = response_scores[key]
            tmp["factoid_scores_"+key] = factoid_scores[key]
        data.update(tmp)
        metadata = {"factoids": self.factoids, "response_fact_questions": response_fact_questions, "response_fact_questions_responses": response_fact_questions_responses, "response_fact_questions_sampled_responses": response_fact_questions_sampled_responses}
        result = {"data": data, "metadata": metadata}
        return UQResult(result)
