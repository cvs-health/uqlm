Code-Generation Scorers
=======================

.. currentmodule:: uqlm.scorers

Code-generation uncertainty quantification uses :class:`CodeGenUQ` to score generated code. These scorers either reuse existing short-form UQ methods or adapt black-box consistency scoring to code by comparing structural similarity or functional equivalence across sampled generations.

.. warning::

   Benchmarking generated code against test cases (e.g., with
   ``uqlm.code.code_evaluation.evaluate_python_code``, as in the
   code-generation demo notebook) executes model-generated code in a
   subprocess with configurable resource limits (memory, CPU time, and file
   size, enforced on POSIX systems) and a wall-clock timeout. This is
   process-level isolation, **not** a security sandbox; do not run untrusted
   code from sources you don't control outside an isolated environment
   (container/VM).

**Key Characteristics:**

- **White-box compatibility:** Token-probability scorers are identical to the corresponding :doc:`white-box scorers <../white_box/index>`.
- **Code-aware consistency:** Black-box scorers compare sampled code generations using code embeddings, CodeBLEU, or LLM-judged functional equivalence.
- **Score range:** :math:`[0, 1]`, where higher values indicate higher confidence.

**Trade-offs:**

- **Dependency requirements:** Some code-aware scorers require code-specific models or language tooling.
- **Higher cost:** Functional equivalence scorers require additional LLM calls.

Code-Generation Scoring Methods
-------------------------------

There are three main categories of code-generation scoring methods offered by UQLM:

.. toctree::
   :maxdepth: 1

   token_probability
   code_similarity
   functional_equivalence
