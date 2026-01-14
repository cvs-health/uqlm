Quickstart Guide
================


Create a virtual environment for using uqlm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend creating a new virtual environment using venv before installing the package. To do so, please follow instructions `here <https://docs.python.org/3/library/venv.html>`_.

Installation
^^^^^^^^^^^^

Install using pip directly from the GitHub repository.

.. code-block:: bash

   pip install uqlm

Usage
-----

Below are minimal examples for hallucination detection.


Example 1: ``Black-Box Scorers`` for hallucination detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These scorers assess uncertainty by measuring the consistency of multiple responses generated from the same prompt. They are compatible with any LLM, intuitive to use, and don't require access to internal model states or token probabilities.

Below is a sample of code illustrating how to use the `BlackBoxUQ` class to conduct hallucination detection.

.. code-block:: python

   from langchain_openai import ChatOpenAI
   llm = ChatOpenAI(model="gpt-4o-mini")

   from uqlm import BlackBoxUQ
   bbuq = BlackBoxUQ(llm=llm, scorers=["semantic_negentropy"], use_best=True)

   results = await bbuq.generate_and_score(prompts=prompts, num_responses=5)
   results.to_df()

.. raw:: html

   <p align="center">
     <img src="./_static/images/black_box_output4.png" />
   </p>

Above, `use_best=True` implements mitigation so that the uncertainty-minimized responses is selected. Note that although we use `ChatOpenAI` in this example, any `LangChain Chat Model <https://js.langchain.com/docs/integrations/chat/>`_  may be used. For a more detailed demo, refer to our `Black-Box UQ Demo <_notebooks/examples/black_box_demo.ipynb>`_.


Example 2: ``White-Box Scorers`` for hallucination detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These scorers leverage token probabilities to estimate uncertainty.  They offer single-generation scoring, which is significantly faster and cheaper than black-box methods, but require access to the LLM's internal probabilities, meaning they are not necessarily compatible with all LLMs/APIs.

Below is a sample of code illustrating how to use the WhiteBoxUQ class to conduct hallucination detection.

.. code-block:: python

   from langchain_google_vertexai import ChatVertexAI
   llm = ChatVertexAI(model='gemini-2.5-pro')

   from uqlm import WhiteBoxUQ
   wbuq = WhiteBoxUQ(llm=llm, scorers=["min_probability"])

   results = await wbuq.generate_and_score(prompts=prompts)
   results.to_df()

.. raw:: html

   <p align="center">
     <img src="./_static/images/white_box_output2.png" />
   </p>

Again, any `LangChain Chat Model <https://js.langchain.com/docs/integrations/chat/>`_ may be used in place of `ChatVertexAI`. For a more detailed demo, refer to our `White-Box UQ Demo <_notebooks/examples/white_box_demo.ipynb>`_.


Example 3: ``LLM-as-a-Judge Scorers`` for hallucination detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These scorers use one or more LLMs to evaluate the reliability of the original LLM's response. They offer high customizability through prompt engineering and the choice of judge LLM(s).

Below is a sample of code illustrating how to use the LLMPanel class to conduct hallucination detection using a panel of LLM judges.

.. code-block:: python

   from langchain_ollama import ChatOllama
   llama = ChatOllama(model="llama3")
   mistral = ChatOllama(model="mistral")
   qwen = ChatOllama(model="qwen3")

   from uqlm import LLMPanel
   panel = LLMPanel(llm=llama, judges=[llama, mistral, qwen])

   results = await panel.generate_and_score(prompts=prompts)
   results.to_df()

.. raw:: html

   <p align="center">
     <img src="./_static/images/panel_output2.png" />
   </p>

Note that although we use `ChatOllama` in this example, we can use any `LangChain Chat Model <https://js.langchain.com/docs/integrations/chat/>`_ as judges. For a more detailed demo, refer to our `LLM-as-a-Judge Demo <_notebooks/examples/judges_demo.ipynb>`_.


Example 4: ``Ensemble Scorers`` for hallucination detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These scorers leverage a weighted average of multiple individual scorers to provide a more robust uncertainty/confidence estimate. They offer high flexibility and customizability, allowing you to tailor the ensemble to specific use cases.

Below is a sample of code illustrating how to use the `UQEnsemble` class to conduct hallucination detection.

.. code-block:: python

   from langchain_openai import AzureChatOpenAI
   llm = AzureChatOpenAI(deployment_name="gpt-4o", openai_api_type="azure", openai_api_version="2024-12-01-preview")

   from uqlm import UQEnsemble
   ## ---Option 1: Off-the-Shelf Ensemble---
   # uqe = UQEnsemble(llm=llm)
   # results = await uqe.generate_and_score(prompts=prompts, num_responses=5)

   ## ---Option 2: Tuned Ensemble---
   scorers = [ # specify which scorers to include
      "exact_match", "noncontradiction", # black-box scorers
      "min_probability", # white-box scorer
      llm # use same LLM as a judge
   ]
   uqe = UQEnsemble(llm=llm, scorers=scorers)

   # Tune on tuning prompts with provided ground truth answers
   tune_results = await uqe.tune(
      prompts=tuning_prompts, ground_truth_answers=ground_truth_answers
   )
   # ensemble is now tuned - generate responses on new prompts
   results = await uqe.generate_and_score(prompts=prompts)
   results.to_df()

.. raw:: html

   <p align="center">
     <img src="./_static/images/uqensemble_output2.png" />
   </p>

As with the other examples, any `LangChain Chat Model <https://js.langchain.com/docs/integrations/chat/>`_ may be used in place of `AzureChatOpenAI`. For more detailed demos, refer to our `Off-the-Shelf Ensemble Demo <_notebooks/examples/ensemble_off_the_shelf_demo.ipynb>`_ (quick start) or our `Ensemble Tuning Demo <_notebooks/examples/ensemble_tuning_demo.ipynb>`_ (advanced).


Example 5: ``Long-Text Scorers`` for hallucination detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These scorers take a fine-grained approach and score confidence/uncertainty at the claim or sentence level. An extension of black-box scorers, long-text scorers sample multiple responses to the same prompt, decompose the original response into claims or sentences, and evaluate consistency of each original claim/sentence with the sampled responses. After scoring claims in the response, the response can be refined by removing claims with confidence scores less than a specified threshold and reconstructing the response from the retained claims. This approach allows for improved factual precision of long-text generations. 

Below is a sample of code illustrating how to use the LongTextUQ class to conduct claim-level hallucination detection and uncertainty-aware response refinement.

.. code-block:: python

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o")

    from uqlm import LongTextUQ
    luq = LongTextUQ(llm=llm, scorers=["entailment"], response_refinement=True)

    results = await luq.generate_and_score(prompts=prompts, num_responses=5)
    results_df = results.to_df()
    results_df

    # Preview the data for a specific claim in the first response
    # results_df["claims_data"][0][0]
    # Output:
    # {
    #   'claim': 'Suthida Bajrasudhabimalalakshana was born on June 3, 1978.',
    #   'removed': False,
    #   'entailment': 0.9548099517822266
    # }

.. raw:: html

   <p align="center">
     <img src="./_static/images/long_text_output.png" />
   </p>

Above `response` and `entailment` reflect the original response and response-level confidence score, while `refined_response` and `refined_entailment` are the corresponding values after response refinement. The `claims_data` column includes granular data for each response, including claims, claim-level confidence scores, and whether each claim is retained in the response refinement process. We use `ChatOpenAI` in this example, any `LangChain Chat Model <https://js.langchain.com/docs/integrations/chat/>`_ may be used. For a more detailed demo, refer to our `Long-Text UQ Demo <_notebooks/examples/long_text_uq_demo.ipynb>`_.


Example notebooks
-----------------
Refer to our :doc:`example notebooks <_notebooks/index>` for examples illustrating how to use `uqlm`.