UQLM FAQs
=========


What is UQLM?
^^^^^^^^^^^^^

UQLM (Uncertainty Quantification for Language Models) is an open-source Python package for generation-time, zero-resource LLM hallucination detection using state-of-the-art uncertainty quantification (UQ) techniques. This toolkit offers a suite of UQ-based scorers to quantify uncertainty in LLM outputs and uniquely integrates generation and evaluation processes. These scorers compute response-level confidence scores between 0 and 1, where higher scores indicate a lower likelihood of errors or hallucinations. These scorers span four categories: black-box UQ (based on consistency of multiple responses to the same prompt), white-box UQ (based on token probabilities), LLM-as-a-Judge (using one or more LLMs to evaluate a question-response concatenation), and ensembles (combining these other scorers).


Why should we care about LLM hallucinations?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Large language models (LLMs) have revolutionized the field of natural language processing, but their tendency to generate false or misleading content, known as hallucinations, significantly compromises safety and trust. LLM hallucinations are especially problematic because they often appear plausible, making them difficult to detect and posing serious risks in high-stakes domains such as healthcare, legal, and financial applications. Even recent models, such as OpenAI’s GPT-4.5, have been found to hallucinate as often as 37.1% on certain benchmarks, underscoring the ongoing challenge of ensuring reliability in LLM outputs. As LLMs are increasingly deployed in real-world settings, monitoring and detecting hallucinations becomes crucial.


How does UQLM differ from other hallucination detection toolkits?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Traditional hallucination detection methods involve ‘grading’ LLM responses by comparing model output to human-authored ground-truth texts, an approach offered by toolkits such as Evals and G-Eval. While effective during pre-deployment/offline testing, these methods are limited in practice since users typically lack access to ground truth data at generation time. This shortcoming motivates the need for generation-time hallucination detection methods.

Existing solutions to this problem include source-comparison methods and uncertainty quantification (UQ) methods. Toolkits that offer source-comparison scorers, such as Ragas, Phoenix, DeepEval, and others, evaluate the consistency between generated content and input prompts. However, these methods can mistakenly validate responses that merely mimic prompt phrasing without ensuring factual accuracy. Although numerous UQ techniques have been proposed in the literature, their adoption in user-friendly, comprehensive toolkits remains limited. Existing UQ-based hallucination detection toolkits tend to be narrow in scope or lack ease of use for non-specialized practitioners.

We aim to bridge this gap with UQLM, which implements a diverse array of uncertainty estimation techniques to compute generation-time, response-level confidence scores and uniquely integrates generation and evaluation processes. This integrated approach allows users to generate and assess content simultaneously, without the need for ground-truth data or external knowledge sources, and with minimal engineering effort. This democratization of access empowers smaller teams, researchers, and developers to incorporate robust hallucination detection into their applications, contributing to the development of safer and more reliable AI systems.


How should I choose among UQLM's scorers?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Choosing the right confidence scorer for an LLM system depends on several factors, including API support, latency requirements, and the availability of graded datasets. First, practitioners should consider whether the API supports access to token probabilities in LLM generations. If so, white-box scorers can be implemented without adding latency or generation costs. However, if the API does not provide access to token probabilities, black-box scorers and LLM-as-a-Judge may be the only feasible options.

When choosing among black-box and LLM-as-a-Judge scorers, latency requirements are a key consideration. For low-latency applications, practitioners should avoid using higher-latency black-box scorers such as semantic negentropy, non-contradiction probability, BERTScore, and BLEURT. Instead, they can opt for faster black-box scorers or use LLM-as-a-Judge. On the other hand, if latency is not a concern, any of the black-box scorers may be used.

Finally, if a graded dataset is available, practitioners can ensemble various confidence scores to improve hallucination detection performance. A graded dataset can be used to tune an ensemble classifier, which can potentially provide more accurate confidence scores than individual scorers. By considering these factors and choosing the right confidence score, practitioners can optimize the performance of their LLM system.


How can UQLM's scorers be used in practice?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Ans) UQLM’s scorers return a confidence score between 0 and 1, where higher scores indicate a lower likelihood of errors or hallucinations. These scores can be utilized in practice in several effective ways:

**Filter or Block Responses with Low Confidence Scores:** By implementing a scoring system that automatically filters out responses that fall below a certain confidence threshold, ensuring that only high-quality, reliable outputs are presented to users.

**Provide ‘Low Confidence’ Disclaimer with Responses:** When a response is generated with a low confidence score, it may be preferrable to include a disclaimer indicating the uncertainty of the response. This transparency helps users understand the reliability of the information provided.

**Targeted Human-in-the-Loop:** For responses that receive low confidence scores, UQLM can facilitate an informed, targeted human-in-the-loop approach. Under this approach, manual review would happen for any response that yields a low confidence score. Additionally, this approach helps address scenarios where exhaustive human-in-the-loop is infeasible due to the scale of responses being generated. Note that selecting a confidence threshold for flagging or blocking responses will depend on the scorer used, the dataset being evaluated, and the stakeholder values (e.g., relative cost of false negatives vs. false positives).

**Pre-Deployment Diagnostics:** UQLM’s scoring system can be employed for pre-deployment diagnostics, allowing developers to assess the performance and reliability of the model in various scenarios. This helps identify potential issues and improve the LLM system’s performance before it is fully deployed.


How does UQLM's ensemble approach work? Is it necessary?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

UQLM offers both tunable and off-the-shelf ensembles that leverage a weighted average of any combination of black-box UQ, white-box UQ, and LLM-as-a-Judge scorers. Using the specified scorers, the ensemble score is computed as a weighted average of the individual confidence scores, where weights may be default weights, user-specified, or tuned.

**Off-the-Shelf Ensemble:** If no scorers are specified, the off-the-shelf implementation follows an ensemble of exact match, non-contradiction probability, and self-judge proposed by `Chen & Mueller, 2023 <https://arxiv.org/abs/2308.16175>`_.

**Tunable Ensemble:** Ensemble tuning enables users to optimize a weighted average of scorers for a specific use case. Note that this approach involves more work and may not be ideal for casual users. To tune the ensemble weights, users must provide a list of prompts and corresponding ideal responses to serve as an ‘answer key’. The LLM’s responses to the prompts are graded with a grader function that compares against the provided ideal responses. If a grader function is not provided by the user, the default grader function that leverages `vectara/hallucination_evaluation_model` is used. Once the binary grades (‘correct’ or ‘incorrect’) are obtained, an optimization routine solves for the optimal weights according to a specified classification objective. The objective function may be threshold-agnostic, such as ROC-AUC, or threshold-dependent, such as F1-score. After completing the optimization routine, the optimized weights are stored as class attributes to be used for subsequent scoring.


Where can I learn more about how UQLM's scorers work under the hood?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A technical description of the UQLM scorers and extensive experiment results are contained in `this paper. <https://arxiv.org/abs/2504.19254>`_
