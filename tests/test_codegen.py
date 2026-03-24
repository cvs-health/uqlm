import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from uqlm.scorers.shortform.codegen import CodeGenUQ
from uqlm.utils.results import UQResult


@pytest.fixture
def mock_llm():
    m = MagicMock()
    m.logprobs = False
    return m


@pytest.fixture
def all_scorers():
    return ["sequence_probability", "min_probability", "mean_token_negentropy", "min_token_negentropy", "probability_margin", "p_true", "consistency_and_confidence", "monte_carlo_probability", "codebleu", "code_equivalence", "verbalized_confidence", "functional_entropy", "semantic_sets", "cosine_sim"]


# ---------- validate_scorers ----------------------------------------------------


@patch("uqlm.scorers.shortform.codegen.WhiteBoxUQ")
@patch("uqlm.scorers.shortform.codegen.CosineScorer")
@patch("uqlm.scorers.shortform.codegen.CodeBLEU")
@patch("uqlm.scorers.shortform.codegen.VerbalizedConfidence")
@patch("uqlm.scorers.shortform.codegen.FunctionalEntropy")
def test_validate_scorers_initializes_components(mock_fe, mock_vc, mock_cb, mock_cos, mock_wb, mock_llm, all_scorers):
    cg = CodeGenUQ(llm=mock_llm, scorers=all_scorers)

    mock_wb.assert_called_once()
    mock_cos.assert_called_once()
    mock_cb.assert_called_once()
    mock_vc.assert_called_once()
    mock_fe.assert_called_once()


# ---------- generate_and_score --------------------------------------------------


@pytest.mark.asyncio
async def test_generate_and_score_calls_dependencies(mock_llm, all_scorers):
    cg = CodeGenUQ(llm=mock_llm, scorers=all_scorers)

    cg.generate_original_responses = AsyncMock(return_value=["A"])
    cg.generate_candidate_responses = AsyncMock(return_value=[["B"]])

    cg.score = AsyncMock(return_value=UQResult(result={"data": {"ok": True}}))

    # mock attributes normally created in parent class
    cg.logprobs = [[-1.0]]
    cg.multiple_logprobs = [[-1.0]]

    result = await cg.generate_and_score(prompts=["test"])

    assert isinstance(result, UQResult)
    cg.generate_original_responses.assert_awaited_once()
    cg.generate_candidate_responses.assert_awaited_once()
    cg.score.assert_awaited_once()


# ---------- score() ------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_produces_expected_data(mock_llm, all_scorers):
    cg = CodeGenUQ(llm=mock_llm, scorers=all_scorers)

    # Mock components
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

    fe_result = UQResult(result={"data": {"discrete_confidence_scores": [0.1], "tokenprob_confidence_scores": [0.2], "num_semantic_sets": [1], "semantic_sets_confidence": [0.3], "cluster_indices": [[0]], "equivalence_rate": [1.0], "original_equivalence_scores": [0.7]}})

    cg.fe = MagicMock()
    cg.fe.evaluate = AsyncMock(return_value=fe_result)
    cg.fe.equivalence_indicators = [1]

    prompts = ["print(1)"]
    responses = ["print(1)"]
    sampled_res = [["print(1)"]]
    logprobs = [[-1.2]]
    sampled_lp = [[-1.1]]

    result = await cg.score(prompts=prompts, responses=responses, sampled_responses=sampled_res, logprobs_results=logprobs, sampled_logprobs_results=sampled_lp)

    assert isinstance(result, UQResult)

    data = result.data

    assert "verbalized_confidence" in data
    assert "cosine_sim" in data
    assert "sequence_probability" in data
    assert "codebleu" in data
    assert "semantic_entropy" in data
    assert "semantic_negentropy" in data
    assert "functional_entropy_equivalence_indicators" in data
    assert "cluster_indices" in data
