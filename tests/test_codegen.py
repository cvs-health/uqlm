import sys
import importlib.util as _importlib_util
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Patch Cosine module before CodeGenUQ is imported (prevents sentence-transformers load)
sys.modules["uqlm.black_box.cosine"] = MagicMock()

from uqlm.scorers.shortform.codegen import CodeGenUQ
from uqlm.utils.results import UQResult

# IMPORT AFTER PATCHING MODULES


# find_spec replacement: returns a fake spec for "codebleu" and delegates
# every other lookup to the real find_spec.
_real_find_spec = _importlib_util.find_spec


def _find_spec_codebleu_only(name, *args, **kwargs):
    if name == "codebleu":
        return MagicMock()
    return _real_find_spec(name, *args, **kwargs)


@pytest.fixture
def mock_llm():
    m = MagicMock()
    m.logprobs = False
    return m


@pytest.fixture
def all_scorers():
    # Lists scorers accepted by _validate_scorers() that the test exercises.
    # Excluded:
    #   - "consistency_and_confidence": _validate_scorers() narrows
    #     self.scorers down to {cosine_sim, sequence_probability} when it is
    #     present, which would mask the rest of the scorer outputs.
    #   - "code_bleu" / "codebleu": tracked separately in upstream issues
    #     (name mismatch between the validator and score() method).
    return ["sequence_probability", "min_probability", "mean_token_negentropy", "min_token_negentropy", "probability_margin", "p_true", "monte_carlo_probability", "cosine_sim", "verbalized_confidence", "functional_negentropy", "functional_sets_confidence", "functional_equivalence_rate"]


# validate_scorers


@patch("importlib.util.find_spec", side_effect=_find_spec_codebleu_only)
@patch("uqlm.code.verbalizedconfidence.VerbalizedConfidence")
@patch("uqlm.code.entropy.FunctionalEntropy")
@patch("uqlm.scorers.shortform.white_box.WhiteBoxUQ")
def test_validate_scorers_initializes_components(mock_wb, mock_fe, mock_vc, mock_find_spec, mock_llm, all_scorers):
    # Patch CodeBLEU inside the test
    with patch.dict(sys.modules, {"codebleu": MagicMock(calc_codebleu=MagicMock(return_value={"codebleu": 0.75}))}):
        cg = CodeGenUQ(llm=mock_llm, scorers=all_scorers)
        assert isinstance(cg, CodeGenUQ)


# generate_and_score


@patch("importlib.util.find_spec", side_effect=_find_spec_codebleu_only)
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


@patch("importlib.util.find_spec", side_effect=_find_spec_codebleu_only)
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
        assert "functional_negentropy" in data


# Regression tests for fixed bugs


@patch("uqlm.scorers.shortform.white_box.WhiteBoxUQ")
def test_validate_scorers_c_and_c_unions_prereqs(mock_wb, mock_llm):
    """Regression: requesting consistency_and_confidence must add cosine_sim + sequence_probability, not replace the set."""
    cg = CodeGenUQ(llm=mock_llm, scorers=["consistency_and_confidence"])
    assert "consistency_and_confidence" in cg.scorers
    assert "cosine_sim" in cg.scorers
    assert "sequence_probability" in cg.scorers


@patch("importlib.util.find_spec", side_effect=_find_spec_codebleu_only)
def test_validate_scorers_accepts_code_bleu_spelling(mock_find_spec, mock_llm):
    """Regression: 'code_bleu' is the documented name; validator must accept it and init self.cb."""
    with patch.dict(sys.modules, {"codebleu": MagicMock(calc_codebleu=MagicMock(return_value={"codebleu": 0.75}))}):
        cg = CodeGenUQ(llm=mock_llm, scorers=["code_bleu"])
        assert "code_bleu" in cg.scorers
        assert hasattr(cg, "cb")


def test_validate_scorers_rejects_unknown(mock_llm):
    """Unknown scorer name must raise ValueError."""
    with pytest.raises(ValueError):
        CodeGenUQ(llm=mock_llm, scorers=["not_a_real_scorer"])


@patch("uqlm.code.entropy.FunctionalEntropy")
def test_validate_scorers_defaults_when_none(mock_fe, mock_llm):
    """scorers=None must fall back to DEFAULT_SCORERS."""
    cg = CodeGenUQ(llm=mock_llm, scorers=None)
    assert set(cg.scorers) == {"functional_equivalence_rate", "cosine_sim"}
