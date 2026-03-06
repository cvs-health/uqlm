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

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pandas as pd
import asyncio
from rich.progress import Progress

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from uqlm.nli.entailment import EntailmentClassifier, SYSTEM_PROMPT, STR_SCORE_MAP


class TestEntailmentClassifier(unittest.TestCase):
    def setUp(self):
        # Create a mock LLM for testing
        self.mock_llm = MagicMock(spec=BaseChatModel)
        self.mock_llm.ainvoke = AsyncMock()
        self.classifier = EntailmentClassifier(nli_llm=self.mock_llm)

    def test_init(self):
        """Test initialization of EntailmentClassifier"""
        self.assertEqual(self.classifier.nli_llm, self.mock_llm)
        self.assertEqual(self.classifier.completed, 0)
        self.assertIsNone(self.classifier.num_responses)

    def test_extract_single_score(self):
        """Test the _extract_single_score method"""
        # Test exact matches at start
        self.assertEqual(self.classifier._extract_single_score("Yes, the claim is entailed."), 1.0)
        self.assertEqual(self.classifier._extract_single_score("No, the claim is not entailed."), 0.0)

        # Test case insensitivity
        self.assertEqual(self.classifier._extract_single_score("yes, definitely entailed"), 1.0)
        self.assertEqual(self.classifier._extract_single_score("NO, not at all"), 0.0)

        # Test substring matches
        self.assertEqual(self.classifier._extract_single_score("I would say yes to this claim"), 1.0)
        self.assertEqual(self.classifier._extract_single_score("The answer is no because..."), 0.0)

        # Test no match
        self.assertTrue(np.isnan(self.classifier._extract_single_score("Maybe, it's unclear")))
        self.assertTrue(np.isnan(self.classifier._extract_single_score("I'm unsure")))

    def test_extract_scores(self):
        """Test the _extract_scores method"""
        responses = ["Yes, the claim is entailed.", "No, the claim is not entailed.", "Uncertain."]
        expected = [1.0, 0.0, np.nan]
        results = self.classifier._extract_scores(responses)

        # Check first two results directly
        self.assertEqual(results[0], expected[0])
        self.assertEqual(results[1], expected[1])
        # Check NaN with isnan
        self.assertTrue(np.isnan(results[2]))

    def test_construct_prompts(self):
        """Test the _construct_prompts method"""
        premises = ["The dog is barking.", "The cat is sleeping."]
        hypotheses = ["The animal is making noise.", "The pet is resting."]

        # Patch the get_entailment_prompt function
        with patch("uqlm.nli.entailment.get_entailment_prompt") as mock_get_prompt:
            mock_get_prompt.side_effect = lambda claim, source_text, style: f"Prompt for {claim} and {source_text}"

            prompts = self.classifier._construct_prompts(premises, hypotheses)

            # Check that the function was called correctly
            self.assertEqual(len(mock_get_prompt.call_args_list), 2)
            mock_get_prompt.assert_any_call(claim=hypotheses[0], source_text=premises[0], style="binary")
            mock_get_prompt.assert_any_call(claim=hypotheses[1], source_text=premises[1], style="binary")

            # Check the returned prompts
            self.assertEqual(prompts[0], f"Prompt for {hypotheses[0]} and {premises[0]}")
            self.assertEqual(prompts[1], f"Prompt for {hypotheses[1]} and {premises[1]}")

    def test_flatten_inputs(self):
        """Test the _flatten_inputs method"""
        response_sets = [["Response 1A", "Response 1B"], ["Response 2A", "Response 2B", "Response 2C"]]
        claim_sets = [["Claim 1X", "Claim 1Y", "Claim 1Z"], ["Claim 2X", "Claim 2Y"]]

        flat_responses, flat_claims, indices, shapes = self.classifier._flatten_inputs(response_sets, claim_sets)

        # Check shapes
        self.assertEqual(shapes[0], (3, 2))  # 3 claims, 2 responses in first set
        self.assertEqual(shapes[1], (2, 3))  # 2 claims, 3 responses in second set

        # Check total number of flattened items
        expected_total = 3 * 2 + 2 * 3  # First set: 3 claims × 2 responses, Second set: 2 claims × 3 responses
        self.assertEqual(len(flat_responses), expected_total)
        self.assertEqual(len(flat_claims), expected_total)
        self.assertEqual(len(indices), expected_total)

        # Check specific items
        # First set, first claim, first response
        self.assertEqual(flat_responses[0], "Response 1A")
        self.assertEqual(flat_claims[0], "Claim 1X")
        self.assertEqual(indices[0], (0, 0, 0))

        # First set, last claim, last response
        idx = 3 * 2 - 1  # Last item in first set
        self.assertEqual(flat_responses[idx], "Response 1B")
        self.assertEqual(flat_claims[idx], "Claim 1Z")
        self.assertEqual(indices[idx], (0, 2, 1))

        # Second set, first claim, first response
        idx = 3 * 2  # First item in second set
        self.assertEqual(flat_responses[idx], "Response 2A")
        self.assertEqual(flat_claims[idx], "Claim 2X")
        self.assertEqual(indices[idx], (1, 0, 0))

    def test_format_result_arrays(self):
        """Test the _format_result_arrays method"""
        # Setup test data
        flat_predictions = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        indices = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (0, 2, 0),
            (0, 2, 1),  # First set: 3 claims × 2 responses
            (1, 0, 0),
            (1, 0, 1),
            (1, 0, 2),
            (1, 1, 0),
            (1, 1, 1),
            (1, 1, 2),  # Second set: 2 claims × 3 responses
        ]
        shapes = [(3, 2), (2, 3)]  # First set: 3×2, Second set: 2×3

        result_matrices = self.classifier._format_result_arrays(flat_predictions, indices, shapes)

        # Check number of matrices
        self.assertEqual(len(result_matrices), 2)

        # Check dimensions of matrices
        self.assertEqual(result_matrices[0].shape, (3, 2))
        self.assertEqual(result_matrices[1].shape, (2, 3))

        # Check values in first matrix
        expected_matrix1 = np.array([[1, 0], [1, 0], [1, 0]])
        np.testing.assert_array_equal(result_matrices[0], expected_matrix1)

        # Check values in second matrix
        expected_matrix2 = np.array([[1, 0, 1], [0, 1, 0]])
        np.testing.assert_array_equal(result_matrices[1], expected_matrix2)

    @patch("uqlm.nli.entailment.get_entailment_prompt")
    async def test_evaluate_claim_response_pair(self, mock_get_prompt):
        """Test the _evaluate_claim_response_pair method"""
        # Setup mock
        self.mock_llm.ainvoke.return_value = AIMessage(content="Yes, the claim is entailed.")

        # Call the method
        result = await self.classifier._evaluate_claim_response_pair("Test prompt")

        # Check that LLM was called with correct messages
        self.mock_llm.ainvoke.assert_called_once()
        call_args = self.mock_llm.ainvoke.call_args[0][0]
        self.assertEqual(len(call_args), 2)
        self.assertEqual(call_args[0].content, SYSTEM_PROMPT)
        self.assertEqual(call_args[1].content, "Test prompt")

        # Check result
        self.assertEqual(result, "Yes, the claim is entailed.")

        # Test with progress bar
        mock_progress = MagicMock(spec=Progress)
        self.classifier.progress_task = "task_id"
        self.classifier.num_prompts = 10
        self.classifier.completed = 0

        await self.classifier._evaluate_claim_response_pair("Test prompt", progress_bar=mock_progress)

        # Check that progress was updated
        mock_progress.update.assert_called_once()

    @patch.object(EntailmentClassifier, "_construct_prompts")
    @patch.object(EntailmentClassifier, "_extract_scores")
    @patch.object(EntailmentClassifier, "_evaluate_claim_response_pair")
    async def test_judge_entailment(self, mock_evaluate, mock_extract, mock_construct):
        """Test the judge_entailment method"""
        # Setup mocks
        mock_construct.return_value = ["Prompt 1", "Prompt 2"]
        mock_evaluate.side_effect = [asyncio.Future(), asyncio.Future()]
        mock_evaluate.side_effect[0].set_result("Yes")
        mock_evaluate.side_effect[1].set_result("No")
        mock_extract.return_value = [1.0, 0.0]

        # Call the method
        result = await self.classifier.judge_entailment(premises=["Premise 1", "Premise 2"], hypotheses=["Hypothesis 1", "Hypothesis 2"])

        # Check that methods were called correctly
        mock_construct.assert_called_once_with(premises=["Premise 1", "Premise 2"], hypotheses=["Hypothesis 1", "Hypothesis 2"])
        self.assertEqual(mock_evaluate.call_count, 2)
        mock_extract.assert_called_once_with(["Yes", "No"])

        # Check result
        expected_result = {"judge_prompts": ["Prompt 1", "Prompt 2"], "judge_responses": ["Yes", "No"], "scores": [1.0, 0.0]}
        self.assertEqual(result, expected_result)

        # Test with retry logic
        mock_extract.side_effect = [[1.0, np.nan], [0.0]]  # First call has a NaN, second call succeeds
        mock_evaluate.reset_mock()
        mock_evaluate.side_effect = [asyncio.Future(), asyncio.Future(), asyncio.Future()]
        mock_evaluate.side_effect[0].set_result("Yes")
        mock_evaluate.side_effect[1].set_result("Unclear")
        mock_evaluate.side_effect[2].set_result("No")

        result = await self.classifier.judge_entailment(premises=["Premise 1", "Premise 2"], hypotheses=["Hypothesis 1", "Hypothesis 2"], retries=1)

        # Check that retry was attempted
        self.assertEqual(mock_evaluate.call_count, 3)  # Initial 2 calls + 1 retry

    @patch.object(EntailmentClassifier, "_flatten_inputs")
    @patch.object(EntailmentClassifier, "judge_entailment")
    @patch.object(EntailmentClassifier, "_format_result_arrays")
    async def test_evaluate_claim_entailment(self, mock_format, mock_judge, mock_flatten):
        """Test the evaluate_claim_entailment method"""
        # Setup mocks
        mock_flatten.return_value = (["Flat Response 1", "Flat Response 2"], ["Flat Claim 1", "Flat Claim 2"], [(0, 0, 0), (0, 1, 0)], [(2, 1)])
        mock_judge.return_value = asyncio.Future()
        mock_judge.return_value.set_result({"judge_prompts": ["Prompt 1", "Prompt 2"], "judge_responses": ["Yes", "No"], "scores": [1.0, 0.0]})
        mock_format.return_value = [np.array([[1.0], [0.0]])]

        # Call the method
        result = await self.classifier.evaluate_claim_entailment(response_sets=[["Response 1"]], claim_sets=[["Claim 1", "Claim 2"]], retries=3, progress_bar=None)

        # Check that methods were called correctly
        mock_flatten.assert_called_once_with(response_sets=[["Response 1"]], claim_sets=[["Claim 1", "Claim 2"]])
        mock_judge.assert_called_once_with(hypotheses=["Flat Claim 1", "Flat Claim 2"], premises=["Flat Response 1", "Flat Response 2"], retries=3, progress_bar=None)
        mock_format.assert_called_once_with(flat_predictions=[1.0, 0.0], indices=[(0, 0, 0), (0, 1, 0)], shapes=[(2, 1)])

        # Check result
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], np.array([[1.0], [0.0]]))

        # Check that num_responses was set
        self.assertEqual(self.classifier.num_responses, 1)
