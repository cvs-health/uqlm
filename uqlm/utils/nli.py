from typing import Any, Optional, Literal, Tuple, Union
import warnings
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from uqlm.utils.prompts.entailment_prompts import get_entailment_prompt


class NLIResult(BaseModel):
    """
    Result from NLI prediction with probabilities.

    This unified model supports both binary and ternary NLI styles.
    The structure adapts based on the `style` field.
    """

    style: Literal["binary", "ternary"] = Field(..., description="The NLI style used")

    # Binary fields (populated when style="binary")
    binary_label: Optional[bool] = Field(None, description="True if entailed, False otherwise (binary style only)")
    binary_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Probability of entailment (binary style only)")

    # Ternary fields (populated when style="ternary")
    ternary_label: Optional[Literal["contradiction", "neutral", "entailment"]] = Field(None, description="Predicted NLI class (ternary style only)")
    ternary_probabilities: Optional[Tuple[float, float, float]] = Field(None, description="Probabilities for [contradiction, neutral, entailment] (ternary style only)")

    @property
    def label(self) -> Union[bool, str]:
        """Get the label regardless of style."""
        if self.style == "binary":
            return self.binary_label
        else:  # ternary
            return self.ternary_label

    @property
    def entailment_probability(self) -> Optional[float]:
        """Get entailment probability regardless of style."""
        if self.style == "binary" and self.binary_probability:
            return self.binary_probability
        elif self.style == "ternary" and self.ternary_probabilities:
            return self.ternary_probabilities[2]
        return None

    @property
    def contradiction_probability(self) -> Optional[float]:
        """Get contradiction probability (ternary only)."""
        if self.style == "ternary" and self.ternary_probabilities:
            return self.ternary_probabilities[0]
        return None

    @property
    def neutral_probability(self) -> Optional[float]:
        """Get neutral probability (ternary only)."""
        if self.style == "ternary" and self.ternary_probabilities:
            return self.ternary_probabilities[1]
        return None


class NLI:
    def __init__(self, nli_model_name: Optional[str] = "microsoft/deberta-large-mnli", nli_llm: Optional[BaseChatModel] = None, max_length: int = 2000, device: Any = None) -> None:
        """
        A class to compute NLI predictions.

        Parameters
        ----------
        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which HuggingFace NLI model to use. Must be acceptable input to
            AutoTokenizer.from_pretrained() and AutoModelForSequenceClassification.from_pretrained().
            Ignored if nli_llm is provided.

        nli_llm : BaseChatModel, default=None
            A LangChain chat model for LLM-based NLI inference. If provided, takes precedence over nli_model_name.

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError. Only applies to HuggingFace models.

        device : torch.device input or torch.device object, default=None
            Specifies the device that classifiers use for prediction. Set to "cuda" for classifiers to be able to
            leverage the GPU. Only applies to HuggingFace models.
        """
        # Prioritize nli_llm if provided, otherwise use nli_model_name
        self.is_hf_model = nli_llm is None
        self.max_length = max_length
        self.label_mapping = ["contradiction", "neutral", "entailment"]
        self._logprobs_warning_shown = False  # Track if we've warned about missing logprobs

        if self.is_hf_model:
            # Initialize HuggingFace model
            if nli_model_name is None:
                raise ValueError("Must specify either nli_model_name or nli_llm.")
            self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
            self.device = device
            self.model = model.to(self.device) if self.device else model
        else:
            # LangChain model
            self.device = None
            self.tokenizer = None
            self.model = nli_llm
            self._activate_logprobs()  # Attempt to activate logprobs

    def predict(self, hypothesis: str, premise: str, style: str = "ternary", return_probabilities: bool = True) -> Any:
        """
        This method computes NLI predictions on the provided inputs.

        Parameters
        ----------
        hypothesis : str
            The hypothesis text for NLI classification.

        premise : str
            The premise text for NLI classification.

        style : str, default="ternary"
            The style of NLI classification to use.
            - "ternary" for ternary classification [contradiction, neutral, entailment]
            - "binary" for binary entailment classification [True, False]

        return_probabilities : bool, default=True
            If True, includes probability information in the result.
            If False, probabilities are set to None.

        Returns
        -------
        NLIResult
            Result object containing label and optionally probabilities.
            Access via result.label or result.binary_label/ternary_label.
        """
        if self.is_hf_model:
            return self._predict_hf(hypothesis, premise, style, return_probabilities)
        else:
            return self._predict_langchain(hypothesis, premise, style, return_probabilities)

    def _predict_hf(self, hypothesis: str, premise: str, style: str = "ternary", return_probabilities: bool = True) -> Any:
        """
        Perform NLI prediction using a HuggingFace model.

        Parameters
        ----------
        hypothesis : str
            The hypothesis text for NLI classification.

        premise : str
            The premise text for NLI classification.

        style : str, default="ternary"
            The NLI style: "ternary" or "binary"

        return_probabilities : bool, default=True
            If True, returns probabilities for all three classes (ternary) or dict with label and probability (binary).
            If False, returns the predicted class label (ternary) or boolean (binary).

        Returns
        -------
        numpy.ndarray or str or bool or dict
            - Ternary + return_probabilities=True: numpy array with 3 probabilities
            - Ternary + return_probabilities=False: string with the predicted class label
            - Binary + return_probabilities=True: dict with {'label': bool, 'probability': float}
            - Binary + return_probabilities=False: boolean
        """
        if len(hypothesis) > self.max_length or len(premise) > self.max_length:
            warnings.warn("Maximum response length exceeded for NLI comparison. Truncation will occur. To adjust, change the value of max_length")

        concat = premise[0 : self.max_length] + " [SEP] " + hypothesis[0 : self.max_length]
        encoded_inputs = self.tokenizer(concat, padding=True, return_tensors="pt")

        if self.device:
            encoded_inputs = {name: tensor.to(self.device) for name, tensor in encoded_inputs.items()}

        logits = self.model(**encoded_inputs).logits
        np_logits = logits.detach().cpu().numpy() if self.device else logits.detach().numpy()
        probabilities = np.exp(np_logits) / np.exp(np_logits).sum(axis=-1, keepdims=True)

        if style == "binary":
            # Binary NLI: True if entailment has highest probability, False otherwise
            predicted_class_idx = probabilities.argmax(axis=1)[0]
            entailment_idx = 2  # entailment is at index 2 in [contradiction, neutral, entailment]
            entailment_prob = float(probabilities[0][entailment_idx])
            label = predicted_class_idx == entailment_idx

            return NLIResult(style="binary", binary_label=bool(label), binary_probability=entailment_prob if return_probabilities else None)
        else:
            # Ternary NLI
            predicted_class_idx = probabilities.argmax(axis=1)[0]
            label = self.label_mapping[predicted_class_idx]
            probs_tuple = tuple(float(p) for p in probabilities[0]) if return_probabilities else None

            return NLIResult(style="ternary", ternary_label=label, ternary_probabilities=probs_tuple)

    def _predict_langchain(self, hypothesis: str, premise: str, style: str = "ternary", return_probabilities: bool = True) -> Any:
        """
        Perform NLI prediction using a LangChain BaseChatModel.

        This method queries the LLM to estimate probabilities or determine class labels.

        Parameters
        ----------
        hypothesis : str
            The hypothesis text (claim) for NLI classification.

        premise : str
            The premise text (source text) for NLI classification.

        style : str, default="ternary"
            The NLI style: "ternary" or "binary"

        return_probabilities : bool, default=True
            If True, estimates probabilities for all classes.
            If False, performs a single query to determine the most likely class.

        Returns
        -------
        numpy.ndarray or str or bool or dict
            Depends on style and return_probabilities settings
        """
        if style == "binary":
            # Binary NLI: Ask if premise entails hypothesis
            prompt = get_entailment_prompt(claim=hypothesis, source_text=premise, style="binary")
            messages = [SystemMessage("You are a helpful assistant that evaluates natural language inference relationships."), HumanMessage(prompt)]

            try:
                response = self.model.invoke(messages)
                response_text = response.content.strip().lower()

                # Determine label from literal text response
                parsed = self._parse_response(response_text, ["yes", "no"])
                label = (parsed == "yes") if parsed is not None else False  # Default to False if unclear

                # Extract probability from logprobs (if available) when requested
                # _extract_yes_probability_from_response already returns probability of entailment
                if return_probabilities:
                    prob = self._extract_yes_probability_from_response(response, response_text)
                else:
                    prob = None

                return NLIResult(style="binary", binary_label=label, binary_probability=prob)
            except Exception as e:
                warnings.warn(f"Error during binary LangChain NLI inference: {e}. Defaulting to False.")
                return NLIResult(style="binary", binary_label=False, binary_probability=0.5 if return_probabilities else None)

        # Ternary style logic
        if return_probabilities:
            # Query the LLM for each class to get probability estimates
            # We'll use Yes/No prompts for each class and interpret the response
            probabilities = []

            for style in ["p_false", "p_neutral", "p_true"]:  # contradiction, neutral, entailment
                prompt = get_entailment_prompt(claim=hypothesis, source_text=premise, style=style)
                messages = [SystemMessage("You are a helpful assistant that evaluates natural language inference relationships."), HumanMessage(prompt)]

                try:
                    response = self.model.invoke(messages)
                    response_text = response.content.strip().lower()

                    # Try to extract probability from logprobs if available
                    prob = self._extract_yes_probability_from_response(response, response_text)

                    probabilities.append(prob)
                except Exception as e:
                    warnings.warn(f"Error during LangChain NLI inference: {e}. Assigning uniform probabilities.")
                    return NLIResult(style="ternary", ternary_label="neutral", ternary_probabilities=(1 / 3, 1 / 3, 1 / 3))

            # Normalize probabilities
            probabilities = np.array(probabilities)
            if probabilities.sum() > 0:
                probabilities = probabilities / probabilities.sum()
            else:
                probabilities = np.array([1 / 3, 1 / 3, 1 / 3])

            # Determine predicted label
            predicted_idx = probabilities.argmax()
            predicted_label = self.label_mapping[predicted_idx]
            probs_tuple = tuple(float(p) for p in probabilities)

            return NLIResult(style="ternary", ternary_label=predicted_label, ternary_probabilities=probs_tuple)

        else:
            # Single query to determine the class (when return_probabilities=False)
            # Use a general entailment prompt that asks for classification
            prompt = get_entailment_prompt(claim=hypothesis, source_text=premise, style="nli_classification")

            messages = [SystemMessage("You are a helpful assistant that evaluates natural language inference relationships."), HumanMessage(prompt)]

            try:
                response = self.model.invoke(messages)
                response_text = response.content.strip().lower()

                # Parse response to get class label
                parsed = self._parse_response(response_text, ["entailment", "contradiction", "neutral"])
                label = parsed if parsed is not None else "neutral"
                if parsed is None:
                    warnings.warn(f"Unclear NLI response from LangChain model: '{response_text}'. Defaulting to 'neutral'.")

                return NLIResult(
                    style="ternary",
                    ternary_label=label,
                    ternary_probabilities=None,  # No probabilities when return_probabilities=False
                )
            except Exception as e:
                warnings.warn(f"Error during LangChain NLI inference: {e}. Defaulting to 'neutral'.")
                return NLIResult(style="ternary", ternary_label="neutral", ternary_probabilities=None)

    def _parse_response(self, response_text: str, expected_values: list) -> Any:
        """
        Parse LLM response to extract one of the expected values.

        Strategy:
        1. Check if response starts with expected value (most reliable)
        2. Check if first word matches expected value
        3. Fallback to full text search (least reliable)

        Parameters
        ----------
        response_text : str
            The response text from the LLM (should already be lowercased and stripped)
        expected_values : list
            List of possible values to look for (e.g., ["yes", "no"] or ["entailment", "contradiction", "neutral"])

        Returns
        -------
        Any
            The matched value from expected_values, or None if no match found
        """
        # Get first word (strip punctuation)
        first_word = response_text.split()[0].strip(".,!?;:") if response_text else ""

        # Try each expected value
        for value in expected_values:
            value_lower = value.lower()

            # Best: response starts with the value
            if response_text.startswith(value_lower):
                return value

            # Good: first word is the value
            if first_word == value_lower:
                return value

        # Fallback: substring search (least reliable, but better than nothing)
        for value in expected_values:
            if value.lower() in response_text:
                return value

        # No match found
        return None

    def _extract_yes_probability_from_response(self, response: Any, response_text: str) -> float:
        """
        Extract the probability of "Yes" from a LangChain response.

        If logprobs are available, uses the actual token probability.
        Otherwise, falls back to binary classification based on response text.

        Supports both OpenAI and Vertex AI (Gemini) logprobs formats.

        Parameters
        ----------
        response : AIMessage or similar
            The response object from the LangChain model
        response_text : str
            The lowercased content of the response

        Returns
        -------
        float
            Probability estimate for "Yes" answer (between 0 and 1)
        """
        # Try to extract from logprobs if available
        if hasattr(response, "response_metadata") and response.response_metadata:
            # Vertex AI (Gemini) style logprobs
            if "logprobs_result" in response.response_metadata:
                logprobs_result = response.response_metadata["logprobs_result"]
                if logprobs_result and len(logprobs_result) > 0:
                    first_token_data = logprobs_result[0]
                    token = first_token_data.get("token", "").strip().lower()
                    logprob = first_token_data.get("logprob", None)

                    if logprob is not None:
                        prob = np.exp(logprob)
                        # Interpret based on what token was generated
                        # Always return probability of "Yes" (entailment)
                        if token.startswith("yes"):
                            return prob  # High prob means high entailment
                        elif token.startswith("no"):
                            return 1.0 - prob  # High prob of "No" means low entailment
                        else:
                            return 0.5  # Unknown token

            # OpenAI-style logprobs
            if "logprobs" in response.response_metadata:
                logprobs_data = response.response_metadata["logprobs"]
                if logprobs_data and "content" in logprobs_data:
                    content_logprobs = logprobs_data["content"]
                    if content_logprobs and len(content_logprobs) > 0:
                        # Get the first token's logprob directly
                        first_token = content_logprobs[0]
                        token = first_token.get("token", "").strip().lower()
                        logprob = first_token.get("logprob", None)

                        if logprob is not None:
                            prob = np.exp(logprob)
                            # Interpret based on what token was generated
                            # Always return probability of "Yes" (entailment)
                            if token.startswith("yes"):
                                return prob  # High prob means high entailment
                            elif token.startswith("no"):
                                return 1.0 - prob  # High prob of "No" means low entailment
                            else:
                                return 0.5  # Unknown token

        # Fallback: binary classification based on text
        # Warn user once if logprobs are not available
        if not self._logprobs_warning_shown and not self.is_hf_model:
            warnings.warn("No logprobs found in LLM response. Probability estimates will be based on response text only (1.0 for 'Yes', 0.0 for 'No', 0.5 for unclear). Ensure your LLM supports logprobs, or set return_probabilities=False if it does not.", UserWarning)
            self._logprobs_warning_shown = True

        # Use robust parsing to determine response
        parsed = self._parse_response(response_text, ["yes", "no"])
        if parsed == "yes":
            return 1.0
        elif parsed == "no":
            return 0.0
        else:
            # If unclear, assign neutral probability
            return 0.5

    # ===== Async Methods =====

    async def apredict(self, hypothesis: str, premise: str, style: str = "ternary", return_probabilities: bool = True) -> Any:
        """
        Async version of predict() for single NLI prediction.

        This method computes NLI predictions on the provided inputs asynchronously.
        For LangChain models, this enables concurrent LLM calls which significantly improves performance.
        For HuggingFace models, this wraps the synchronous call for API consistency.

        Parameters
        ----------
        hypothesis : str
            The hypothesis text for NLI classification.

        premise : str
            The premise text for NLI classification.

        style : str, default="ternary"
            The style of NLI classification to use.
            - "ternary" for ternary classification [contradiction, neutral, entailment]
            - "binary" for binary entailment classification [True, False]

        return_probabilities : bool, default=True
            If True, returns probabilities for all classes.
            If False, returns a single class label.

        Returns
        -------
        numpy.ndarray or str or bool or dict
            Same as predict() method
        """
        if self.is_hf_model:
            # HF models are synchronous, so we just call the sync method
            # Wrapped in an async context for API consistency
            return self._predict_hf(hypothesis, premise, style, return_probabilities)
        else:
            return await self._apredict_langchain(hypothesis, premise, style, return_probabilities)

    async def _apredict_langchain(self, hypothesis: str, premise: str, style: str = "ternary", return_probabilities: bool = True) -> Any:
        """
        Async version of _predict_langchain() using LangChain's async interface.

        This method uses ainvoke() and asyncio.gather() to make concurrent LLM calls,
        which can significantly reduce latency (up to 3x faster when return_probabilities=True).

        Parameters
        ----------
        hypothesis : str
            The hypothesis text (claim) for NLI classification.

        premise : str
            The premise text (source text) for NLI classification.

        style : str, default="ternary"
            The NLI style: "ternary" or "binary"

        return_probabilities : bool, default=True
            If True, estimates probabilities for all classes.
            If False, performs a single query to determine the most likely class.

        Returns
        -------
        numpy.ndarray or str or bool or dict
            Depends on style and return_probabilities settings
        """
        if style == "binary":
            # Binary NLI: Ask if premise entails hypothesis
            prompt = get_entailment_prompt(claim=hypothesis, source_text=premise, style="binary")
            messages = [SystemMessage("You are a helpful assistant that evaluates natural language inference relationships."), HumanMessage(prompt)]

            try:
                response = await self.model.ainvoke(messages)
                response_text = response.content.strip().lower()

                # Determine label from literal text response
                parsed = self._parse_response(response_text, ["yes", "no"])
                label = (parsed == "yes") if parsed is not None else False  # Default to False if unclear

                # Extract probability from logprobs (if available) when requested
                # _extract_yes_probability_from_response already returns probability of entailment
                if return_probabilities:
                    prob = self._extract_yes_probability_from_response(response, response_text)
                else:
                    prob = None

                return NLIResult(style="binary", binary_label=label, binary_probability=prob)
            except Exception as e:
                warnings.warn(f"Error during async binary LangChain NLI inference: {e}. Defaulting to False.")
                return NLIResult(style="binary", binary_label=False, binary_probability=0.5 if return_probabilities else None)

        if return_probabilities:
            # Query the LLM for each class concurrently to get probability estimates
            async def query_style(style: str) -> float:
                """Query a single style and return probability."""
                prompt = get_entailment_prompt(claim=hypothesis, source_text=premise, style=style)
                messages = [SystemMessage("You are a helpful assistant that evaluates natural language inference relationships."), HumanMessage(prompt)]

                try:
                    response = await self.model.ainvoke(messages)
                    response_text = response.content.strip().lower()
                    prob = self._extract_yes_probability_from_response(response, response_text)
                    return prob
                except Exception as e:
                    warnings.warn(f"Error during async LangChain NLI inference for style '{style}': {e}")
                    return None

            # Execute all three queries concurrently
            try:
                prob_results = await asyncio.gather(query_style("p_false"), query_style("p_neutral"), query_style("p_true"))

                # Check if any queries failed
                if None in prob_results:
                    warnings.warn("One or more async NLI queries failed. Assigning uniform probabilities.")
                    return NLIResult(style="ternary", ternary_label="neutral", ternary_probabilities=(1 / 3, 1 / 3, 1 / 3))

                probabilities = np.array(prob_results)

                # Normalize probabilities
                if probabilities.sum() > 0:
                    probabilities = probabilities / probabilities.sum()
                else:
                    probabilities = np.array([1 / 3, 1 / 3, 1 / 3])

                # Determine predicted label
                predicted_idx = probabilities.argmax()
                predicted_label = self.label_mapping[predicted_idx]
                probs_tuple = tuple(float(p) for p in probabilities)

                return NLIResult(style="ternary", ternary_label=predicted_label, ternary_probabilities=probs_tuple)

            except Exception as e:
                warnings.warn(f"Error during async LangChain NLI inference: {e}. Assigning uniform probabilities.")
                return NLIResult(style="ternary", ternary_label="neutral", ternary_probabilities=(1 / 3, 1 / 3, 1 / 3))

        else:
            # Single query to determine the class (when return_probabilities=False)
            prompt = get_entailment_prompt(claim=hypothesis, source_text=premise, style="nli_classification")
            messages = [SystemMessage("You are a helpful assistant that evaluates natural language inference relationships."), HumanMessage(prompt)]

            try:
                response = await self.model.ainvoke(messages)
                response_text = response.content.strip().lower()

                # Parse response to get class label
                parsed = self._parse_response(response_text, ["entailment", "contradiction", "neutral"])
                label = parsed if parsed is not None else "neutral"
                if parsed is None:
                    warnings.warn(f"Unclear NLI response from async LangChain model: '{response_text}'. Defaulting to 'neutral'.")

                return NLIResult(
                    style="ternary",
                    ternary_label=label,
                    ternary_probabilities=None,  # No probabilities when return_probabilities=False
                )
            except Exception as e:
                warnings.warn(f"Error during async LangChain NLI inference: {e}. Defaulting to 'neutral'.")
                return NLIResult(style="ternary", ternary_label="neutral", ternary_probabilities=None)

    def _activate_logprobs(self) -> None:
        """
        Attempt to activate logprobs for the LLM.
        """
        if self.is_hf_model:
            warnings.warn("Logprobs are not supported for HuggingFace models. Please use a LangChain model instead.")
            return
        if hasattr(self.model, "logprobs"):
            self.model.logprobs = True
        else:
            warnings.warn("Logprobs are not supported for this model. Please use a model that supports logprobs.")
