import pytest
import os
import platform
import sys
from unittest.mock import MagicMock, AsyncMock, patch
from uqlm.scorers.shortform.codegen import CodeGenUQ
from uqlm.utils.results import UQResult

pytestmark = pytest.mark.skipif((os.getenv("CI") == "true" and platform.system() == "Linux") or platform.system() == "Windows", reason="Skipping transformer-heavy tests on CI Linux and Windows")

#  Patch Cosine module before CodeGenUQ is imported
cosine_mock = MagicMock()
sys.modules["uqlm.black_box.cosine"] = cosine_mock  # Prevent importing sentence-transformers

# IMPORT AFTER PATCHING MODULES


@pytest.fixture
def mock_llm():
    m = MagicMock()
    m.logprobs = False
    return m


@pytest.fixture
def all_scorers():
    return ["sequence_probability", "min_probability", "mean_token_negentropy", "min_token_negentropy", "probability_margin", "p_true", "consistency_and_confidence", "monte_carlo_probability", "codebleu", "code_equivalence", "verbalized_confidence", "functional_entropy", "semantic_sets", "cosine_sim"]


# validate_scorers


@patch("importlib.util.find_spec", return_value=True)
@patch("uqlm.code.verbalizedconfidence.VerbalizedConfidence")
@patch("uqlm.code.entropy.FunctionalEntropy")
@patch("uqlm.scorers.shortform.white_box.WhiteBoxUQ")
def test_validate_scorers_initializes_components(mock_wb, mock_fe, mock_vc, mock_find_spec, mock_llm, all_scorers):
    # Patch CodeBLEU inside the test
    with patch.dict(sys.modules, {"codebleu": MagicMock(calc_codebleu=MagicMock(return_value={"codebleu": 0.75}))}):
        cg = CodeGenUQ(llm=mock_llm, scorers=all_scorers)
        assert isinstance(cg, CodeGenUQ)


# generate_and_score


@patch("importlib.util.find_spec", return_value=True)
@patch("uqlm.scorers.shortform.white_box.WhiteBoxUQ")
@pytest.mark.asyncio
async def test_generate_and_score_calls_dependencies(mock_wb, mock_find_spec, mock_llm, all_scorers):
    with patch.dict(sys.modules, {"codebleu": MagicMock(calc_codebleu=MagicMock(return_value={"codebleu": 0.75}))}):
        cg = CodeGenUQ(llm=mock_llm, scorers=all_scorers)

        cg.generate_original_responses = AsyncMock(return_value=["A"])
        cg.generate_candidate_responses = AsyncMock(return_value=[["B"]])
        cg.score = AsyncMock(return_value=UQResult(result={"data": {"ok": True}}))

        cg.logprobs = [[-1.0]]
        cg.multiple_logprobs = [[-1.0]]

        result = await cg.generate_and_score(prompts=["test"])

        assert isinstance(result, UQResult)
        cg.generate_original_responses.assert_awaited_once()
        cg.generate_candidate_responses.assert_awaited_once()
        cg.score.assert_awaited_once()


# score()


@patch("importlib.util.find_spec", return_value=True)
@patch("uqlm.scorers.shortform.white_box.WhiteBoxUQ")
@pytest.mark.asyncio
async def test_score_produces_expected_data(mock_wb, mock_find_spec, mock_llm, all_scorers):
    with patch.dict(sys.modules, {"codebleu": MagicMock(calc_codebleu=MagicMock(return_value={"codebleu": 0.75}))}):
        cg = CodeGenUQ(llm=mock_llm, scorers=all_scorers)

        cg.vc = MagicMock()
        cg.vc.judge_responses = AsyncMock(return_value=[0.5])

        cg.cos = MagicMock()
        cg.cos.evaluate = MagicMock(return_value=[0.9])
        cg.cos.pair_scores = [0.99]

        cg.cb = MagicMock()
        cg.cb.evaluate = MagicMock(return_value=[0.8])
        cg.cb.pair_scores = [0.88]

        cg.wbuq_scorers = ["sequence_probability"]
        cg.wbuq = MagicMock()
        cg.wbuq.score = AsyncMock(return_value=MagicMock(data={"sequence_probability": [0.4]}))

        fe_result = {"functional_negentropy": [0.1], "functional_negentropy_whitebox": [0.2], "functional_sets_confidence": [0.3], "functional_equivalence_rate": [1.0]}

        cg.fe = MagicMock()
        cg.fe.evaluate = AsyncMock(return_value=fe_result)
        cg.fe.equivalence_indicators = [1]

        result = await cg.score(prompts=["print(1)"], responses=["print(1)"], sampled_responses=[["print(1)"]], logprobs_results=[[-1.2]], sampled_logprobs_results=[[[-1.1]]])

        data = result.data

        assert "verbalized_confidence" in data
        assert "cosine_sim" in data
        assert "sequence_probability" in data
        assert "codebleu" in data
        assert "functional_negentropy" in data
