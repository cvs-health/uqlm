from typing import Any
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from uqlm.utils.prompt_templates import get_binary_entailment_template
from langchain_core.language_models.chat_models import BaseChatModel

class NLI:
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

    def predict(self, hypothesis: str, premise: str, return_probabilities: bool = True) -> Any:
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
        if return_probabilities:
            return probabilites
        else:
            if self.label_mapping[probabilites.argmax()] == "entailment":
                return True
            else:
                return False

class EntailmentLLMJudge:
    def __init__(self, 
                 llm: BaseChatModel,
                 ) -> None:
        self.llm = llm

    async def predict(self, hypothesis: str | list[str], premise: str | list[str]) -> Any:
        """
        This method compute probability of contradiction on the provide inputs.
        """
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis]
        if isinstance(premise, str):
            premise = [premise]
        prompts = []
        results = []
        for h,p in zip(hypothesis,premise):
            prompts.append(get_binary_entailment_template(h, p))
        responses = await self.llm.ainvoke(prompts)
        for res in responses:
            if "true" in res.content.lower():
                results.append(True)
            elif "false" in res.content.lower():
                results.append(False)
            else:
                results.append(None)
        return results
