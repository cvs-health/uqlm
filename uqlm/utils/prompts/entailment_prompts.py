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


def get_entailment_prompt(claim: str, source_text: str, style: str) -> str:
    """
    This is the entailment prompt from GraphUQ.

    Parameters
    ----------
    claim: str
        The claim to be evaluated.
    source_text: str
        The source text to be evaluated.

    Returns
    -------
    str
        The prompt template for evaluating the entailment of a claim and a source text.
    """

    entailment_prompt = None

    if style == "binary":  # this is modified version ofthe "edge construction prompt" from https://arxiv.org/pdf/2410.20783
        entailment_prompt = f"""
        Context: {source_text}
        Claim: {claim}
        Is the claim entailed by the context above?
        Answer Yes or No:
        """

    elif style == "p_true":
        entailment_prompt = f"""
        Source text:
        {source_text}
        Claim:
        {claim}

        Is the claim supported by the source text? 
        
        If so, answer "Yes". Otherwise, if the claim is simply neutral/subjective in regards to the source text or is contradicted by the source text, answer "No".
        
        Answer Yes or No:
        """

    elif style == "p_false":
        entailment_prompt = f"""
        Source text:
        {source_text}
        Claim:
        {claim}

        Is the claim contradicted by the source text? 
        
        If so, answer "Yes". Otherwise, if the claim is simply neutral/subjective in regards to the source text or is supported by the source text, answer "No".
        
        Answer Yes or No:
        """

    elif style == "p_neutral":
        entailment_prompt = f"""
        Source text:
        {source_text}
        Claim:
        {claim}

        Is the claim neutral to the source text? Meaning it is neither supported nor contradicted by the source text. 
        
        If the claim is neutral, answer "Yes". 
        
        If the claim is supported or contradicted by the source text, answer "No".

        Answer Yes or No:
        """

    elif style == "nli_classification":
        entailment_prompt = f"""
        
        Given the following premise and hypothesis, determine the natural language inference relationship.

        Premise: {source_text}

        Hypothesis: {claim}

        What is the relationship between the premise and hypothesis?
        - "entailment" if the hypothesis is supported by the premise
        - "contradiction" if the hypothesis contradicts the premise  
        - "neutral" if the hypothesis is neither supported nor contradicted by the premise

        Answer with only one word: entailment, contradiction, or neutral."""

    else:
        entailment_prompt = f"""

        You are a helpful assistant that can evaluate the entailment of a claim and a source text.

        You will be given a claim and a source text. You need to evaluate if the claim is entailed by the source text.

        You should return one of the following categorizations:

        true - if the claim is entailed by the source text.
        false - if the claim is not entailed by the source text.

        Example:

        Source text:
        Emory University has a strong athletics program, competing in the National Collegiate Athletic Association (NCAA) Division I Atlantic Coast Conference (ACC). The university's mascot is the Eagle.

        Claim:
        Emory University is part of the ACC.

        Categorization:
        true

        Only return the categorization label (true or false). Do not include any other text.

        Source text:
        {source_text}

        Claim:
        {claim}

        Categorization:
        """

    return entailment_prompt
