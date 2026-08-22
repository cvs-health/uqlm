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
import warnings
import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging

from uqlm.utils.device import get_best_device
from uqlm.utils.warn import beta_warning

logging.set_verbosity_error()

# Fallback when neither the tokenizer nor the model config declares a usable token limit
DEFAULT_MAX_TOKENS = 512
# Tokenizers report a very large sentinel (~1e30) for model_max_length when unset
_MODEL_MAX_LENGTH_SENTINEL = 100_000


class NLI:
    def __init__(self, device: Any = None, verbose: bool = False, nli_model_name: str = "microsoft/deberta-large-mnli", max_length: int = 2000, batch_size: int = 32, use_fp16: bool = False, device_map: Optional[str] = None) -> None:
        """
        A class to computing NLI-based confidence scores. This class offers two types of confidence scores, namely
        noncontradiction probability :footcite:`chen2023quantifyinguncertaintyanswerslanguage` and semantic entropy
        :footcite:`farquhar2024detectinghallucinations`.

        Parameters
        ----------
        device : torch.device input or torch.device object, default=None
            Specifies the device that classifiers use for prediction. Set to "cuda" for classifiers to be able to
            leverage the GPU.

        verbose : bool, default=False
            Specifies whether to print verbose status updates of NLI scoring process

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError. Inputs are additionally truncated at the token level to the model's maximum
            sequence length.

        batch_size : int, default=32
            Number of premise-hypothesis pairs scored per forward pass in `predict_batch`. Lower this value if
            inference runs out of memory; raise it to increase GPU utilization.

        use_fp16 : bool, default=False
            If True and the device is CUDA or MPS, runs inference in half precision (torch.float16) to reduce
            memory usage and increase throughput at a small cost to numerical precision. Ignored on CPU.

        device_map : str, default=None
            Optional device map (e.g. "auto") passed to AutoModelForSequenceClassification.from_pretrained() to
            shard or place the model across available devices. Requires the `accelerate` package. If provided,
            takes precedence over `device`. This option is in beta and may change in future releases.
        """
        # Handle device detection
        if device is None:
            device = get_best_device()
        elif isinstance(device, str):
            device = torch.device(device)

        self.verbose = verbose
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        if device_map is not None:
            try:
                import accelerate  # noqa: F401
            except ImportError:
                raise ImportError("The `device_map` option requires the `accelerate` package. Install it with `pip install accelerate` or `pip install 'uqlm[accelerate]'`.")
            beta_warning("The `device_map` option (accelerate integration) is in beta. Please use with caution as it may change in future releases.")
            model = AutoModelForSequenceClassification.from_pretrained(nli_model_name, device_map=device_map)
            device = model.device
        else:
            model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
            model = model.to(device) if device else model
        if use_fp16 and device is not None and getattr(device, "type", None) in ("cuda", "mps"):
            model = model.half()
        self.device = device
        self.model = model.eval()
        self.max_tokens = self._resolve_max_tokens()
        self.label_mapping = ["contradiction", "neutral", "entailment"]
        self.probabilities = dict()

    def predict(self, premise: str, hypothesis: str) -> Any:
        """
        This method compute probability of contradiction on the provide inputs.

        Parameters
        ----------
        premise : str
            An input for the sequence classification DeBERTa model.

        hypothesis : str
            An input for the sequence classification DeBERTa model.

        Returns
        -------
        numpy.ndarray
            Probabilities computed by NLI model
        """
        return self.predict_batch([(premise, hypothesis)])

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> Any:
        """
        Compute NLI probabilities for a list of (premise, hypothesis) pairs using batched inference.

        Parameters
        ----------
        pairs : list of (str, str) tuples
            Premise-hypothesis pairs to score. Pairs are scored in chunks of `batch_size` per forward pass.

        Returns
        -------
        numpy.ndarray
            Array of shape (len(pairs), 3) containing [contradiction, neutral, entailment] probabilities,
            one row per input pair.
        """
        if len(pairs) == 0:
            return np.empty((0, len(self.label_mapping)))
        if any(len(premise) > self.max_length or len(hypothesis) > self.max_length for premise, hypothesis in pairs):
            warnings.warn("Maximum response length exceeded for NLI comparison. Truncation will occur. To adjust, change the value of max_length")
        probabilities = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            premises = [premise[0 : self.max_length] for premise, _ in chunk]
            hypotheses = [hypothesis[0 : self.max_length] for _, hypothesis in chunk]
            encoded_inputs = self.tokenizer(premises, hypotheses, padding=True, truncation="longest_first", max_length=self.max_tokens, return_tensors="pt")
            if self.device:
                encoded_inputs = {name: tensor.to(self.device) for name, tensor in encoded_inputs.items()}
            with torch.no_grad():
                logits = self.model(**encoded_inputs).logits
            np_logits = logits.float().cpu().numpy()
            probabilities.append(np.exp(np_logits) / np.exp(np_logits).sum(axis=-1, keepdims=True))
        return np.concatenate(probabilities, axis=0)

    def get_nli_results(self, response1: str, response2: str) -> Dict[str, Any]:
        """This method computes mean NLI score and determines whether entailment exists."""
        return self.get_nli_results_batch([(response1, response2)])[0]

    def get_nli_results_batch(self, response_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Compute mean NLI scores and entailment indicators for a list of response pairs. Both directions of
        each pair are evaluated in a single batched forward pass.

        Parameters
        ----------
        response_pairs : list of (str, str) tuples
            Response pairs to score.

        Returns
        -------
        list of dict
            One dictionary per input pair with keys "noncontradiction_score", "entailment", and "entailment_score".
        """
        results = [None] * len(response_pairs)
        pending_indices, pending_pairs = [], []
        for i, (response1, response2) in enumerate(response_pairs):
            if response1 == response2:
                results[i] = {"noncontradiction_score": 1, "entailment": True, "entailment_score": 1}
            elif (response1, response2) in self.probabilities and (response2, response1) in self.probabilities:
                results[i] = self._nli_results_from_probabilities(left=self.probabilities[(response1, response2)], right=self.probabilities[(response2, response1)])
            else:
                pending_indices.append(i)
                pending_pairs.extend([(response1, response2), (response2, response1)])
        if pending_indices:
            probabilities = self.predict_batch(pending_pairs)
            for k, i in enumerate(pending_indices):
                response1, response2 = response_pairs[i]
                left, right = probabilities[2 * k : 2 * k + 1], probabilities[2 * k + 1 : 2 * k + 2]
                self.probabilities.update({(response1, response2): left, (response2, response1): right})
                results[i] = self._nli_results_from_probabilities(left=left, right=right)
        return results

    def _nli_results_from_probabilities(self, left: Any, right: Any) -> Dict[str, Any]:
        """Compute mean NLI scores and entailment indicator from both directions' probabilities"""
        left_label = self.label_mapping[left.argmax(axis=1)[0]]
        right_label = self.label_mapping[right.argmax(axis=1)[0]]
        s1, s2 = 1 - left[:, 0], 1 - right[:, 0]
        entailment = left_label == "entailment" or right_label == "entailment"
        return {"noncontradiction_score": ((s1 + s2) / 2)[0], "entailment": entailment, "entailment_score": ((left[:, -1] + right[:, -1]) / 2)[0]}

    def _resolve_max_tokens(self) -> int:
        """Determine the token-level truncation limit from the tokenizer or model config"""
        max_tokens = getattr(self.tokenizer, "model_max_length", None)
        if not max_tokens or max_tokens > _MODEL_MAX_LENGTH_SENTINEL:
            max_tokens = getattr(self.model.config, "max_position_embeddings", 0) or DEFAULT_MAX_TOKENS
        return int(max_tokens)
