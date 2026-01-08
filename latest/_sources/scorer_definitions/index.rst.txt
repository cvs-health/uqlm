Scorer Definitions
==================

This section provides formal mathematical definitions for all uncertainty quantification scorers available in UQLM.
Each scorer returns a confidence score between 0 and 1, where higher scores indicate a lower likelihood of errors or hallucinations.

For detailed API documentation and usage examples, see the :doc:`API Reference </api>` and :doc:`Example Notebooks </_notebooks/index>`.

.. toctree::
   :maxdepth: 2
   :caption: Available Scorers

   black_box/index
   white_box/index
   llm_judges/index
   ensemble/index
   long_text/index

