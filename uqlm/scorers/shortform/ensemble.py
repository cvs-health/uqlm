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
import inspect
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import time
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
import rich
from rich import print as rprint

from uqlm.judges.judge import LLMJudge
from uqlm.scorers.shortform.baseclass.uncertainty import ShortFormUQ
from uqlm.utils.results import UQResult
from uqlm.scorers.shortform.panel import LLMPanel
from uqlm.scorers.shortform.black_box import BlackBoxUQ
from uqlm.scorers.shortform.white_box import WhiteBoxUQ
from uqlm.utils.grader import LLMGrader
from uqlm.utils.tuner import Tuner
from uqlm.utils.llm_config import save_llm_config, load_llm_config

# Define scorer categories to avoid circular imports
TOP_LOGPROBS_SCORER_NAMES = ["min_token_negentropy", "mean_token_negentropy", "probability_margin"]
SAMPLED_LOGPROBS_SCORER_NAMES = ["semantic_negentropy", "semantic_density", "monte_carlo_probability", "consistency_and_confidence"]


class UQEnsemble(ShortFormUQ):
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        scorers: Optional[List[Union[str, BaseChatModel, LLMJudge]]] = None,
        device: Any = None,
        postprocessor: Any = None,
        system_prompt: Optional[str] = None,
        max_calls_per_min: Optional[int] = None,
        use_n_param: bool = False,
        thresh: float = 0.5,
        weights: List[float] = None,
        nli_model_name: str = "microsoft/deberta-large-mnli",
        use_best: bool = True,
        sampling_temperature: float = 1.0,
        scoring_templates: Optional[List[str]] = None,
        max_length: int = 2000,
        verbose: bool = False,
        grader_llm: Optional[BaseChatModel] = None,
        return_responses: str = "all",
    ) -> None:
        """
        Class for detecting bad and speculative answer from a pretrained Large Language Model (LLM Hallucination).

        Parameters
        ----------
        llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        scorers : List[Union[str, BaseChatModel]] default=None
            Specifies which UQ components to include. List containing instances of BaseChatModel, LLMJudge, black-box scorer names from
            ['semantic_negentropy', 'noncontradiction','exact_match', 'bert_score', 'cosine_sim'], or white-box scorer names from
            ["sequence_probability", "min_probability", "min_token_negentropy", "mean_token_negentropy", "probability_margin",
            "semantic_negentropy", "semantic_density", "monte_carlo_probability", "consistency_and_confidence", "p_true"].
            If None, defaults to the off-the-shelf BS Detector ensemble by Chen and Mueller (2023) :footcite:`chen2023quantifyinguncertaintyanswerslanguage`
            which uses components ["noncontradiction", "exact_match", llm] with respective weights of [0.56, 0.14, 0.3]

        device : str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. Only applies to 'semantic_negentropy', 'noncontradiction'
            scorers. If None, detects and returns the best available PyTorch device. Prioritizes CUDA (NVIDIA GPU), then MPS (macOS), then CPU.

        postprocessor : callable, default=None
            A user-defined function that takes a string input and returns a string. Used for postprocessing
            outputs before black-box comparisons.

        use_best : bool, default=True
            Specifies whether to swap the original response for the uncertainty-minimized response
            based on semantic entropy clusters.

        sampling_temperature : float, default=1.0
            The 'temperature' parameter for llm model to generate sampled LLM responses. Must be greater than 0.

        system_prompt : str, default=None
            Optional argument for user to provide custom system prompt. If prompts are list of strings and system_prompt is None,
            defaults to "You are a helpful assistant."

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.

        use_n_param : bool, default=False
            Specifies whether to use `n` parameter for `BaseChatModel`. Not compatible with all
            `BaseChatModel` classes. If used, it speeds up the generation process substantially when num_responses > 1.

        weights : list of floats, default=None
            Specifies weight for each component in ensemble. If None and `scorers` is not None, each component will
            receive equal weight. If `scorers` is None, defaults to the off-the-shelf BS Detector ensemble by
            Chen and Mueller (2023) :footcite:`chen2023quantifyinguncertaintyanswerslanguage` which uses components
            ["noncontradiction", "exact_match","self_reflection"] with respective weights of [0.56, 0.14, 0.3].

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        scoring_templates : List[str], default=None
             Specifies which off-the-shelf template to use for each judge. Four off-the-shelf templates offered:
             incorrect/uncertain/correct (0/0.5/1), incorrect/correct (0/1), continuous score (0 to 1), and likert scale score ( 1-5 scale, normalized to 0/0.25/0.5/0.75/1).
             These templates are respectively specified as 'true_false_uncertain', 'true_false', 'continuous', and 'likert'
             If specified, must be of equal length to `judges` list. Defaults to 'true_false_uncertain' template
             used by Chen and Mueller (2023) :footcite:`chen2023quantifyinguncertaintyanswerslanguage` for each judge.

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError

        return_responses : str, default="all"
            If a postprocessor is used, specifies whether to return only postprocessed responses, only raw responses,
            or both. Specified with 'postprocessed', 'raw', or 'all', respectively.

        verbose : bool, default=False
            Specifies whether to print the index of response currently being scored.
        """
        super().__init__(llm=llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param, postprocessor=postprocessor)
        self.nli_model_name = nli_model_name
        self.thresh = thresh
        self.weights = weights
        self.verbose = verbose
        self.sampling_temperature = sampling_temperature
        self.use_best = use_best
        self.max_length = max_length
        self.return_responses = return_responses
        self.scoring_templates = scoring_templates
        self.tuner = Tuner()
        self._validate_components(scorers)
        self._validate_weights()
        self.grader_llm = llm if not grader_llm else grader_llm

    async def generate_and_score(self, prompts: List[Union[str, List[BaseMessage]]], num_responses: int = 5, show_progress_bars: Optional[bool] = True, _existing_progress_bar: Optional[rich.progress.Progress] = None) -> UQResult:
        """
        Generate LLM responses from provided prompts and compute confidence scores.

        Parameters
        ----------
        prompts : List[Union[str, List[BaseMessage]]]
            List of prompts from which LLM responses will be generated. Prompts in list may be strings or lists of BaseMessage. If providing
            input type List[List[BaseMessage]], refer to https://python.langchain.com/docs/concepts/messages/#langchain-messages for support.
            Must be list of strings if including LLM judges in ensemble.

        num_responses : int, default=5
            The number of sampled responses used to compute consistency.

        show_progress_bars : bool, default=True
            If True, displays progress bars while generating and scoring responses.
            Not displayed for white box scorers due to low latency.

        Returns
        -------
        UQResult
            Instance of UQResult, containing data (prompts, responses, and scores) and metadata
        """
        self.num_responses = num_responses
        if self.white_box_components:
            assert hasattr(self.llm, "logprobs"), """
            In order to use white-box components, BaseChatModel must have logprobs attribute
            """
            self.llm.logprobs = True

        if self.judges:
            if not all(isinstance(item, str) for item in prompts):
                raise ValueError("prompts must be list of strings when using LLM judges with UQEnsemble")
        self._construct_progress_bar(show_progress_bars, _existing_progress_bar=_existing_progress_bar)
        self._display_generation_header(show_progress_bars)

        # Determine if we need top_k_logprobs for top-logprobs scorers
        top_k_logprobs = None
        if self.white_box_components:
            if any(scorer in TOP_LOGPROBS_SCORER_NAMES for scorer in self.white_box_components):
                top_k_logprobs = 15

        responses = await self.generate_original_responses(prompts, top_k_logprobs=top_k_logprobs, progress_bar=self.progress_bar)

        # Generate sampled responses if needed by black-box or sampled-logprobs white-box scorers
        needs_sampled_responses = self.black_box_components or (self.white_box_components and any(scorer in SAMPLED_LOGPROBS_SCORER_NAMES for scorer in self.white_box_components))
        if needs_sampled_responses:
            sampled_responses = await self.generate_candidate_responses(prompts, num_responses=self.num_responses, progress_bar=self.progress_bar)
        else:
            sampled_responses = None
            self.multiple_logprobs = [[None] * self.num_responses] * len(prompts)

        result = await self.score(prompts=prompts, responses=responses, sampled_responses=sampled_responses, logprobs_results=self.logprobs, show_progress_bars=show_progress_bars, _existing_progress_bar=_existing_progress_bar)

        self._stop_progress_bar(_existing_progress_bar)  # if re-run ensure the same progress object is not used
        return result

    async def score(self, prompts: List[str], responses: List[str], sampled_responses: Optional[List[List[str]]] = None, logprobs_results: Optional[List[List[Dict[str, Any]]]] = None, num_responses: int = 5, show_progress_bars: Optional[bool] = True, _existing_progress_bar: Optional[rich.progress.Progress] = None) -> UQResult:
        """
        Generate LLM responses from provided prompts and compute confidence scores.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        responses : list of str
            A list of model responses for the prompts.

        sampled_responses : list of list of str, default=None
            A list of lists of sampled LLM responses for each prompt. These will be used to compute consistency scores by comparing to
            the corresponding response from `responses`. Must be provided if using black box scorers.

        logprobs_results : list of logprobs_result, default=None
            List of lists of dictionaries, each returned by BaseChatModel.ainvoke. Must be provided if using white box scorers.

        num_responses : int, default=5
            The number of sampled responses used to compute consistency. Not value will not be used if sampled_responses is provided

        show_progress_bars : bool, default=True
            If True, displays a progress bar while scoring responses

        Returns
        -------
        UQResult
            Instance of UQResult, containing data (prompts, responses, and semantic entropy scores) and
            metadata
        """
        if self.black_box_components and not sampled_responses:
            raise ValueError("sampled_responses must be provided if using black-box scorers")
        if self.white_box_components and not logprobs_results:
            raise ValueError("logprobs_results must be provided if using white-box scorers")

        self._construct_progress_bar(show_progress_bars, _existing_progress_bar=_existing_progress_bar)
        self._display_scoring_header(show_progress_bars)

        self.prompts = prompts
        self.responses = responses
        self.sampled_responses = sampled_responses
        self.num_responses = num_responses if not sampled_responses else len(sampled_responses[0])
        if not logprobs_results:
            self.logprobs = [None] * len(prompts)
            self.multiple_logprobs = [[None] * self.num_responses] * len(prompts)

        if self.black_box_components:
            self.black_box_object.progress_bar = self.progress_bar
            black_box_results = self.black_box_object.score(responses=self.responses, sampled_responses=self.sampled_responses, show_progress_bars=show_progress_bars, _display_header=False)
            if self.use_best:
                self._update_best(black_box_results.data["responses"])

        if self.white_box_components:
            white_box_results = await self.white_box_object.score(logprobs_results=self.logprobs, prompts=prompts, responses=self.responses, sampled_responses=self.sampled_responses, sampled_logprobs_results=self.multiple_logprobs, show_progress_bars=False)

        if self.judges:
            self._start_progress_bar()
            self.judges_object.progress_bar = self.progress_bar
            judge_results = await self.judges_object.score(prompts=prompts, responses=self.responses, show_progress_bars=show_progress_bars, _display_header=False)
        self.component_scores = {k: [] for k in self.component_names}

        for i, component in enumerate(self.component_scores):
            if component in self.black_box_components:
                self.component_scores[component] = black_box_results.data[component]
            elif component in self.white_box_components:
                self.component_scores[component] = white_box_results.data[component]
            elif i in self.judges_indices:
                self.component_scores[component] = judge_results.data[component]

        self._stop_progress_bar(_existing_progress_bar)  # if re-run ensure the same progress object is not used

        return self._construct_result()

    def tune_from_graded(self, correct_indicators: List[bool], weights_objective: str = "roc_auc", thresh_bounds: Tuple[float, float] = (0, 1), thresh_objective: str = "fbeta_score", n_trials: int = 100, step_size: float = 0.01, fscore_beta: float = 1, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Tunes weights and threshold parameters on a set of user-provided graded responses.

        Parameters
        ----------
        correct_indicators : list of bool
            A list of boolean indicators of whether self.responses are correct.

        weights_objective : {'fbeta_score', 'accuracy_score', 'balanced_accuracy_score', 'roc_auc', 'log_loss', 'average_precision', 'brier_score'}, default='roc_auc'
            Objective function for weight optimization. Must match thresh_objective if one of 'fbeta_score',
            'accuracy_score', 'balanced_accuracy_score'. If same as thresh_objective, joint optimization will be done.

        thresh_bounds : tuple of floats, default=(0,1)
            Bounds to search for threshold

        thresh_objective : {'fbeta_score', 'accuracy_score', 'balanced_accuracy_score'}, default='fbeta_score'
            Objective function for threshold optimization via grid search.

        n_trials : int, default=100
            Indicates how many candidates to search over with optuna optimizer

        step_size : float, default=0.01
            Indicates step size in grid search, if used

        fscore_beta : float, default=1
            Value of beta in fbeta_score

        show_progress_bars : bool, default=True
            If True, displays a progress bar while optimizing weights and threshold

        Returns
        -------
        UQResult
        """
        assert self.component_scores, """
        evaluate method must be run prior to running tune_params method
        """
        score_lists = list(self.component_scores.values())
        optimal_params = self.tuner.tune_params(score_lists=score_lists, correct_indicators=correct_indicators, weights_objective=weights_objective, thresh_bounds=thresh_bounds, thresh_objective=thresh_objective, n_trials=n_trials, step_size=step_size, fscore_beta=fscore_beta, progress_bar=self.progress_bar)
        self.weights = optimal_params["weights"]
        self.thresh = optimal_params["thresh"]
        self._stop_progress_bar()
        self.print_ensemble_weights()
        return self._construct_result()

    async def tune(self, prompts: List[str], ground_truth_answers: List[str], grader_function: Optional[Any] = None, num_responses: int = 5, weights_objective: str = "roc_auc", thresh_bounds: Tuple[float, float] = (0, 1), thresh_objective: str = "fbeta_score", n_trials: int = 100, step_size: float = 0.01, fscore_beta: float = 1, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Generate responses from provided prompts, grade responses with provided grader function, and tune ensemble weights.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        ground_truth_answers : list of str
            A list of ideal (correct) responses

        grader_function : function(response: str, answer: str) -> bool, default=None
            A user-defined function that takes a response and a ground truth 'answer' and returns a boolean indicator of whether
            the response is correct. If not provided, vectara's HHEM is used: https://huggingface.co/vectara/hallucination_evaluation_model

        num_responses : int, default=5
            The number of sampled responses used to compute consistency.

        weights_objective : {'fbeta_score', 'accuracy_score', 'balanced_accuracy_score', 'roc_auc', 'log_loss', 'average_precision', 'brier_score'}, default='roc_auc'
            Objective function for weight optimization. Must match thresh_objective if one of 'fbeta_score',
            'accuracy_score', 'balanced_accuracy_score'. If same as thresh_objective, joint optimization will be done.

        thresh_bounds : tuple of floats, default=(0,1)
            Bounds to search for threshold

        thresh_objective : {'fbeta_score', 'accuracy_score', 'balanced_accuracy_score'}, default='fbeta_score'
            Objective function for threshold optimization via grid search.

        n_trials : int, default=100
            Indicates how many trials to search over with optuna optimizer

        step_size : float, default=0.01
            Indicates step size in grid search, if used

        fscore_beta : float, default=1
            Value of beta in fbeta_score

        show_progress_bars : bool, default=True
            If True, displays a progress bar while while generating responses, scoring responses, and tuning weights/threshold

        Returns
        -------
        UQResult
        """
        self._validate_grader(grader_function)
        self._construct_progress_bar(show_progress_bars)
        await self.generate_and_score(prompts=prompts, num_responses=num_responses, show_progress_bars=show_progress_bars, _existing_progress_bar=self.progress_bar)

        self._start_progress_bar()
        self._display_optimization_header(show_progress_bars)
        correct_indicators = await self._grade_responses(ground_truth_answers=ground_truth_answers, grader_function=grader_function)
        tuned_result = self.tune_from_graded(correct_indicators=correct_indicators, weights_objective=weights_objective, thresh_bounds=thresh_bounds, thresh_objective=thresh_objective, n_trials=n_trials, step_size=step_size, fscore_beta=fscore_beta, show_progress_bars=show_progress_bars)
        self._stop_progress_bar()
        return tuned_result

    def print_ensemble_weights(self):
        """Prints ensemble weights in a pretty table format, sorted by weight in descending order"""
        weights_df = pd.DataFrame({"Scorer": self.component_names, "Weight": self.weights})
        weights_df = weights_df.sort_values(by="Weight", ascending=False)
        weights_df["Weight"] = weights_df["Weight"].apply(lambda x: f"{x:.4f}")

        rprint(" ")

        title = "[bold]Optimized Ensemble Weights"
        centered_title = title.center(50)  # Center the title
        rprint(centered_title)
        rprint("=" * 50)

        header = f"{weights_df.columns[0]:<25}{weights_df.columns[1]:>15}"
        rprint(header)
        rprint("-" * 50)

        for _, row in weights_df.iterrows():
            rprint(f"{row['Scorer']:<25}{row['Weight']:>15}")
        rprint("=" * 50)

    def save_config(self, path: str) -> None:
        """
        Save minimal configuration: weights, threshold, components, and LLM configs.

        Parameters
        ----------
        path : str
            Path where to save the configuration file (should end with .json)
        """

        # Handle components and LLM scorers
        serializable_components = []
        llm_configs = {}
        llm_count = 0

        for component in self.components:
            if isinstance(component, str):
                serializable_components.append(component)
            elif isinstance(component, (LLMJudge, BaseChatModel)):
                llm_count += 1
                llm_key = f"judge_{llm_count}"
                serializable_components.append(llm_key)
                llm_configs[llm_key] = save_llm_config(component)
            else:
                raise ValueError(f"Cannot serialize component: {component}")

        # Save main LLM config if present
        main_llm_config = None
        if self.llm:
            main_llm_config = save_llm_config(self.llm)

        config = {"weights": self.weights, "thresh": self.thresh, "components": serializable_components, "llm_config": main_llm_config, "llm_scorers": llm_configs}

        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_config(cls, path: str, llm: Optional[BaseChatModel] = None) -> "UQEnsemble":
        """
        Load configuration and create UQEnsemble instance.

        Parameters
        ----------
        path : str
            Path to the saved configuration file
        llm : BaseChatModel, optional
            LLM instance to use as main LLM. If None, uses saved config.

        Returns
        -------
        UQEnsemble
            New UQEnsemble instance with loaded configuration
        """
        with open(path, "r") as f:
            config = json.load(f)

        # Recreate main LLM
        if llm is None and config.get("llm_config"):
            llm = load_llm_config(config["llm_config"])

        # Recreate component scorers
        components = []
        llm_scorers = config.get("llm_scorers", {})

        for component in config["components"]:
            if isinstance(component, str) and component.startswith("judge_"):
                # This is an LLM scorer
                if component in llm_scorers:
                    llm_scorer = load_llm_config(llm_scorers[component])
                    components.append(llm_scorer)
                else:
                    raise ValueError(f"Missing LLM config for {component}")
            else:
                # This is a named scorer
                components.append(component)

        return cls(llm=llm, scorers=components, weights=config["weights"], thresh=config["thresh"])

    async def _grade_responses(self, ground_truth_answers: List[str], grader_function: Any = None) -> List[Any]:
        """Grade LLM responses against provided ground truth answers using provided grader function"""
        if grader_function:
            correct_indicators = []
            if self.progress_bar:
                progress_task = self.progress_bar.add_task("  - Grading responses against provided ground truth answers...", total=len(ground_truth_answers))
            for r, a in zip(self.responses, ground_truth_answers):
                correct_indicators.append(grader_function(r, a))
                if self.progress_bar:
                    self.progress_bar.update(progress_task, advance=1)
            time.sleep(0.1)
        else:
            llm_grader = LLMGrader(llm=self.grader_llm)
            correct_indicators = await llm_grader.grade_responses(prompts=self.prompts, responses=self.responses, answers=ground_truth_answers, progress_bar=self.progress_bar)
        return correct_indicators

    def _construct_result(self) -> Any:
        """Constructs UQResult from dictionary"""
        if self.black_box_components:
            data = self._construct_black_box_return_data()
        else:
            data = {"prompts": self.prompts, "responses": self.responses, "sampled_responses": self.sampled_responses if self.sampled_responses else [None] * len(self.responses)}
        data["ensemble_scores"] = self._compute_ensemble_scores(score_dict=self.component_scores, weights=self.weights)
        data.update(self.component_scores)
        result = {"data": data, "metadata": {"temperature": None if not self.llm else self.llm.temperature, "sampling_temperature": None if not self.sampling_temperature else self.sampling_temperature, "num_responses": self.num_responses, "thresh": self.thresh, "weights": self.weights, "logprobs": self.logprobs}}
        return UQResult(result)

    def _compute_ensemble_scores(self, score_dict: Dict[str, List[float]], weights: List[float]):
        """Compute dot product of component scores and weights"""
        score_lists = [np.array(score_dict[key]) for key in score_dict.keys()]
        return self.tuner._compute_ensemble_scores(weights=np.array(weights), score_lists=score_lists).tolist()

    def _validate_components(self, components: List[Any]) -> None:
        "Validate components and construct applicable scorer attributes"
        self.black_box_components, self.white_box_components, self.judges = [], [], []
        self.black_box_indices, self.white_box_indices, self.judges_indices = [], [], []
        self.component_names = []
        if components is None:
            # Default to BS Detector
            components = ["noncontradiction", "exact_match", self.llm]
            self.black_box_components = ["noncontradiction", "exact_match"]
            self.judges.append(self.llm)
            self.component_names = self.black_box_components + ["judge_1"]
            self.judges_indices = [2]
            self.weights = [0.7 * 0.8, 0.7 * 0.2, 0.3]  # Default BS Detector weights
        else:
            judge_count = 0
            for i, component in enumerate(components):
                if component in self.white_box_names:
                    self.white_box_components.append(component)
                    self.white_box_indices.append(i)
                    self.component_names.append(component)
                elif component in self.black_box_names:
                    self.black_box_components.append(component)
                    self.black_box_indices.append(i)
                    self.component_names.append(component)
                elif isinstance(component, (LLMJudge, BaseChatModel)):
                    judge_count += 1
                    self.judges.append(component)
                    self.judges_indices.append(i)
                    self.component_names.append(f"judge_{judge_count}")
                else:
                    raise ValueError(
                        f"""
                        Components must be an instance of LLMJudge, BaseChatModel, a black-box scorer from {self.black_box_names}, or a white-box scorer from {self.white_box_names}
                        """
                    )
        if self.black_box_components:
            self.black_box_object = BlackBoxUQ(scorers=self.black_box_components, device=self.device, nli_model_name=self.nli_model_name, max_length=self.max_length, use_best=self.use_best)
        if self.white_box_components:
            self.white_box_object = WhiteBoxUQ(llm=self.llm, scorers=self.white_box_components, device=self.device, system_prompt=self.system_prompt, max_calls_per_min=self.max_calls_per_min, sampling_temperature=self.sampling_temperature, use_n_param=self.use_n_param, max_length=self.max_length)
        if self.judges:
            self.judges_object = LLMPanel(judges=self.judges, max_calls_per_min=self.max_calls_per_min, scoring_templates=self.scoring_templates)
        self.components = components

    def _validate_weights(self) -> None:
        """Validate ensemble weights"""
        if self.weights:
            if len(self.weights) != len(self.components):
                raise ValueError("Must have same number of weights as components")
            self.weights = self._normalize_weights(self.weights)
        else:
            self.weights = [1 / len(self.components)] * len(self.components)

    def _normalize_weights(self, weights: List[float]) -> List[float]:
        """Normalize weights to sum to 1."""
        weights = weights if weights else [1] * len(self.components)
        return list(self.tuner._normalize_weights(weights))

    @staticmethod
    def _validate_grader(grader_function) -> bool:
        "Validate that grader function is valid"
        if grader_function is None:
            pass
        else:
            sig = inspect.signature(grader_function)
            params = sig.parameters
            if "response" not in params or "answer" not in params:
                raise ValueError("grader_function must have 'response' and 'answer' parameters")
            try:
                check_val = grader_function(response="a", answer="b")
            except Exception:
                check_val = grader_function(response="a", answer=["b"])
            if not isinstance(check_val, bool):
                raise ValueError("grader_function must return boolean")
