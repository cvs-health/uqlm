LLM-as-a-Judge Scorers
======================

LLM-as-a-Judge scorers use one or more LLMs to evaluate the reliability of the original LLM's response.
They offer high customizability through prompt engineering and the choice of judge LLM(s).

**Key Characteristics:**

- **Universal Compatibility:** Works with any LLM
- **Highly Customizable:** Use any LLM as a judge and tailor instruction prompts for specific use cases
- **Self-Reflection Capable:** Can use the same LLM as both generator and judge

**Trade-offs:**

- **Added Cost:** Requires additional LLM calls for the judge LLM(s)
- **Added Latency:** Judge evaluations add to the total response time

**Overview:**

Under the LLM-as-a-Judge approach, either the same LLM that was used for generating the original
responses or a different LLM is asked to form a judgment about a pre-generated response. Several
scoring templates are available to accommodate different use cases.

.. toctree::
   :maxdepth: 1
   :caption: Scoring Templates

   true_false_uncertain
   true_false
   continuous
   likert

Panel of Judges
---------------

For improved robustness, you can use the :class:`~uqlm.scorers.LLMPanel` class to aggregate scores
from multiple LLM judges using various aggregation methods (average, min, max, median).

.. toctree::
   :maxdepth: 1

   panel

