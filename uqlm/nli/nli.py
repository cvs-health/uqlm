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
from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging

from uqlm.utils.device import get_best_device

logging.set_verbosity_error()


class NLI:
    def __init__(self, device: Any = None, verbose: bool = False, nli_model_name: str = "microsoft/deberta-large-mnli", max_length: int = 2000) -> None:
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
            avoid OutOfMemoryError
        """
        # Handle device detection
        if device is None:
            device = get_best_device()
        elif isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.verbose = verbose
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.model = model.to(self.device) if self.device else model
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
        if len(premise) > self.max_length or len(hypothesis) > self.max_length:
            warnings.warn("Maximum response length exceeded for NLI comparison. Truncation will occur. To adjust, change the value of max_length")
        concat = premise[0 : self.max_length] + " [SEP] " + hypothesis[0 : self.max_length]
        encoded_inputs = self.tokenizer(concat, padding=True, return_tensors="pt")
        if self.device:
            encoded_inputs = {name: tensor.to(self.device) for name, tensor in encoded_inputs.items()}
        logits = self.model(**encoded_inputs).logits
        np_logits = logits.detach().cpu().numpy() if self.device else logits.detach().numpy()
        probabilites = np.exp(np_logits) / np.exp(np_logits).sum(axis=-1, keepdims=True)
        return probabilites

    def get_nli_results(self, response1: str, response2: str) -> Dict[str, Any]:
        """This method computes mean NLI score and determines whether entailment exists."""
        if response1 == response2:
            avg_noncontradiction_score, entailment, avg_entailment_score = 1, True, 1
        else:
            left = self.predict(premise=response1, hypothesis=response2)
            left_label = self.label_mapping[left.argmax(axis=1)[0]]

            right = self.predict(premise=response2, hypothesis=response1)
            right_label = self.label_mapping[right.argmax(axis=1)[0]]
            s1, s2 = 1 - left[:, 0], 1 - right[:, 0]

            entailment = left_label == "entailment" or right_label == "entailment"
            avg_noncontradiction_score = ((s1 + s2) / 2)[0]
            avg_entailment_score = ((left[:, -1] + right[:, -1]) / 2)[0]
            self.probabilities.update({f"{response1}_{response2}": left, f"{response2}_{response1}": right})
        return {"noncontradiction_score": avg_noncontradiction_score, "entailment": entailment, "entailment_score": avg_entailment_score}
