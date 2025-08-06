from typing import Any
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


class NLI():
    def __init__(self,
                 nli_model_name: str = "microsoft/deberta-large-mnli",
                 max_length: int = 2000,
                 device: Any = None,
                 ) -> None:
        """
        A class to compute NLI predictions.

        Parameters
        ----------
        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError
        
        device : torch.device input or torch.device object, default=None
            Specifies the device that classifiers use for prediction. Set to "cuda" for classifiers to be able to
            leverage the GPU.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.device = device
        self.max_length = max_length
        self.model = model.to(self.device) if self.device else model
        self.label_mapping = ["contradiction", "neutral", "entailment"]

    def predict(self, hypothesis: str, premise: str) -> Any:
        """
        This method compute probability of contradiction on the provide inputs.

        Parameters
        ----------
        hypothesis : str
            An input for the sequence classification DeBERTa model.

        premise : str
            An input for the sequence classification DeBERTa model.

        Returns
        -------
        numpy.ndarray
            Probabilities computed by NLI model for each label
        """
        if len(hypothesis) > self.max_length or len(premise) > self.max_length:
            warnings.warn("Maximum response length exceeded for NLI comparison. Truncation will occur. To adjust, change the value of max_length")
        concat = hypothesis[0 : self.max_length] + " [SEP] " + premise[0 : self.max_length]
        encoded_inputs = self.tokenizer(concat, padding=True, return_tensors="pt")
        if self.device:
            encoded_inputs = {name: tensor.to(self.device) for name, tensor in encoded_inputs.items()}
        logits = self.model(**encoded_inputs).logits
        np_logits = logits.detach().cpu().numpy() if self.device else logits.detach().numpy()
        probabilites = np.exp(np_logits) / np.exp(np_logits).sum(axis=-1, keepdims=True)
        return probabilites