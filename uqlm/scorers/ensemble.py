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
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from uqlm.judges.judge import LLMJudge
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier, UQResult
from uqlm.scorers.panel import LLMPanel
from uqlm.scorers.black_box import BlackBoxUQ
from uqlm.scorers.white_box import WhiteBoxUQ
from uqlm.utils.tuner import Tuner
from uqlm.utils.llm_config import save_llm_config, load_llm_config


class UQEnsemble(UncertaintyQuantifier):
    def __init__(
        self,
        llm=None,
        scorers: Optional[List[Union[str, BaseChatModel, LLMJudge]]] = None,
        device: Any = None,
        postprocessor: Any = None,
        system_prompt: str = "You are a helpful assistant.",
        max_calls_per_min: Optional[int] = None,
        use_n_param: bool = False,
        thresh: float = 0.5,
        weights: List[float] = None,
        nli_model_name: str = "microsoft/deberta-large-mnli",
        use_best: bool = True,
        sampling_temperature: float = 1.0,
        max_length: int = 2000,
        verbose: bool = False,
    ) -> None:
        """
        Class for detecting bad and speculative answer from a pretrained Large Language Model (LLM Hallucination).

        Parameters
        ----------
        llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        scorers : List containing instances of BaseChatModel, LLMJudge, black-box scorer names from ['semantic_negentropy', 'noncontradiction','exact_match', 'bert_score', 'cosine_sim'], or white-box scorer names from ["normalized_probability", "min_probability"] default=None
            Specifies which UQ components to include. If None, defaults to the off-the-shelf BS Detector ensemble by
            Chen and Mueller (2023) :footcite:`chen2023quantifyinguncertaintyanswerslanguage` which uses components
            ["noncontradiction", "exact_match","self_reflection"] with respective weights of [0.56, 0.14, 0.3]

        device : str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. Only applies to 'semantic_negentropy', 'noncontradiction'
            scorers. Pass a torch.device to leverage GPU.

        postprocessor : callable, default=None
            A user-defined function that takes a string input and returns a string. Used for postprocessing
            outputs.

        use_best : bool, default=True
            Specifies whether to swap the original response for the uncertainty-minimized response
            based on semantic entropy clusters.

        sampling_temperature : float, default=1.0
            The 'temperature' parameter for llm model to generate sampled LLM responses. Must be greater than 0.

        system_prompt : str or None, default="You are a helpful assistant."
            Optional argument for user to provide custom system prompt

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

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError

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
        self.tuner = Tuner()
        self._validate_components(scorers)
        self._validate_weights()

    async def generate_and_score(self, prompts: List[str], num_responses: int = 5, progress_bar: Optional[bool] = True):
        """
        Generate LLM responses from provided prompts and compute confidence scores.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        num_responses : int, default=5
            The number of sampled responses used to compute consistency.

        progress_bar : bool, default=True
            If True, displays a progress bar while generating and scoring responses

        Returns
        -------
        UQResult
            Instance of UQResult, containing data (prompts, responses, and semantic entropy scores) and
            metadata
        """
        self.num_responses = num_responses
        if self.white_box_components:
            assert hasattr(self.llm, "logprobs"), """
            In order to use white-box components, BaseChatModel must have logprobs attribute
            """
            self.llm.logprobs = True
        if progress_bar:
            rprint("🤖 Generation")
        responses = await self.generate_original_responses(prompts, progress_bar=progress_bar)
        if self.black_box_components:
            sampled_responses = await self.generate_candidate_responses(prompts, progress_bar=progress_bar)
        else:
            sampled_responses = None

        return await self.score(prompts=prompts, responses=responses, sampled_responses=sampled_responses, logprobs_results=self.logprobs, progress_bar=progress_bar)

    async def score(self, prompts: List[str], responses: List[str], sampled_responses: Optional[List[List[str]]] = None, logprobs_results: Optional[List[List[Dict[str, Any]]]] = None, num_responses: int = 5, progress_bar: Optional[bool] = True):
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
            List of lists of dictionaries, each returned by BaseChatModel.agenerate. Must be provided if using white box scorers.

        num_responses : int, default=5
            The number of sampled responses used to compute consistency. Not value will not be used if sampled_responses is provided

        progress_bar : bool, default=True
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

        self.prompts = prompts
        self.responses = responses
        self.sampled_responses = sampled_responses
        self.num_responses = num_responses if not sampled_responses else len(sampled_responses[0])
        if not logprobs_results:
            self.logprobs = [None] * len(prompts)
            self.multiple_logprobs = [[None] * self.num_responses] * len(prompts)

        if progress_bar:
            rprint("📈 Scoring")
        if self.black_box_components:
            black_box_results = self.black_box_object.score(responses=self.responses, sampled_responses=self.sampled_responses, progress_bar=progress_bar)
            if self.use_best:
                self._update_best(black_box_results.data["responses"])

        if self.white_box_components:
            white_box_results = self.white_box_object.score(logprobs_results=self.logprobs)

        if self.judges:
            judge_results = await self.judges_object.score(prompts=prompts, responses=self.responses, progress_bar=progress_bar)
        self.component_scores = {k: [] for k in self.component_names}

        for i, component in enumerate(self.component_scores):
            if component in self.black_box_components:
                self.component_scores[component] = black_box_results.data[component]
            elif component in self.white_box_components:
                self.component_scores[component] = white_box_results.data[component]
            elif i in self.judges_indices:
                self.component_scores[component] = judge_results.data[component]

        return self._construct_result()

    def tune_from_graded(self, correct_indicators: List[bool], weights_objective: str = "roc_auc", thresh_bounds: Tuple[float, float] = (0, 1), thresh_objective: str = "fbeta_score", n_trials: int = 100, step_size: float = 0.01, fscore_beta: float = 1, progress_bar: Optional[bool] = True) -> UQResult:
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

        Returns
        -------
        UQResult
        """
        assert self.component_scores, """
        evaluate method must be run prior to running tune_params method
        """
        score_lists = list(self.component_scores.values())
        optimal_params = self.tuner.tune_params(score_lists=score_lists, correct_indicators=correct_indicators, weights_objective=weights_objective, thresh_bounds=thresh_bounds, thresh_objective=thresh_objective, n_trials=n_trials, step_size=step_size, fscore_beta=fscore_beta, progress_bar=progress_bar)
        self.weights = optimal_params["weights"]
        self.thresh = optimal_params["thresh"]

        self.print_ensemble_weights()
        return self._construct_result()

    async def tune(self, prompts: List[str], ground_truth_answers: List[str], grader_function: Optional[Any] = None, num_responses: int = 5, weights_objective: str = "roc_auc", thresh_bounds: Tuple[float, float] = (0, 1), thresh_objective: str = "fbeta_score", n_trials: int = 100, step_size: float = 0.01, fscore_beta: float = 1, progress_bar: Optional[bool] = True) -> UQResult:
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

        progress_bar : bool, default=True
            If True, displays a progress bar while while generating responses, scoring responses, and tuning weights

        Returns
        -------
        UQResult
        """
        self._validate_grader(grader_function)
        await self.generate_and_score(prompts=prompts, num_responses=num_responses, progress_bar=progress_bar)

        if progress_bar:
            rprint("⚙️ Optimization")
        correct_indicators = self._grade_responses(ground_truth_answers=ground_truth_answers, grader_function=grader_function, progress_bar=progress_bar)
        tuned_result = self.tune_from_graded(correct_indicators=correct_indicators, weights_objective=weights_objective, thresh_bounds=thresh_bounds, thresh_objective=thresh_objective, n_trials=n_trials, step_size=step_size, fscore_beta=fscore_beta, progress_bar=progress_bar)
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

    def _grade_responses(self, ground_truth_answers: List[str], grader_function: Any = None, progress_bar: bool = True) -> List[Any]:
        """Grade LLM responses against provided ground truth answers using provided grader function"""
        if grader_function:
            correct_indicators = []
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.completed}/{task.total}"), TimeElapsedColumn()) as progress:
                if progress_bar:
                    progress_task = progress.add_task("- [black]Grading responses against provided ground truth answers...", total=len(ground_truth_answers))
                for r, a in zip(self.responses, ground_truth_answers):
                    correct_indicators.append(grader_function(r, a))
                    if progress_bar:
                        progress.update(progress_task, advance=1)
                time.sleep(0.1)
        else:
            self._construct_hhem()  # use vectara hhem if no grader is provided
            pairs = [(a, r) for a, r in zip(ground_truth_answers, self.responses)]
            if progress_bar:
                rprint("  - [black]Grading responses against provided ground truth answers...")
            halluc_scores = self.hhem.predict(pairs)
            correct_indicators = [(s > 0.5) * 1 for s in halluc_scores]
        return correct_indicators

    def _construct_result(self) -> Any:
        """Constructs UQResult from dictionary"""
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
            self.white_box_object = WhiteBoxUQ()
        if self.judges:
            self.judges_object = LLMPanel(judges=self.judges)
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

    def _construct_hhem(self):
        from transformers import AutoModelForSequenceClassification

        self.hhem = AutoModelForSequenceClassification.from_pretrained("vectara/hallucination_evaluation_model", trust_remote_code=True)

    @staticmethod
    def _validate_grader(grader_function: Any) -> bool:
        "Validate that grader function is valid"
        if grader_function is None:
            pass
        else:
            sig = inspect.signature(grader_function)
            params = sig.parameters
            if "response" not in params or "answer" not in params:
                raise ValueError("grader_function must have 'response' and 'answer' parameters")
            check_val = grader_function("a", "b")
            if not isinstance(check_val, bool):
                raise ValueError("grader_function must return boolean")
