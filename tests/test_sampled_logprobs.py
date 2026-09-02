import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from uqlm.white_box.sampled_logprobs import SampledLogprobsScorer, SAMPLED_LOGPROBS_SCORER_NAMES


@pytest.fixture
def scorer():
    """Fixture to create a SampledLogprobsScorer instance."""
    mock_llm = MagicMock()
    return SampledLogprobsScorer(llm=mock_llm)


def test_initialization(scorer):
    """Test the initialization of SampledLogprobsScorer."""
    assert scorer.scorers == SAMPLED_LOGPROBS_SCORER_NAMES
    assert scorer.llm is not None
    assert scorer.nli_model_name == "microsoft/deberta-large-mnli"
    assert scorer.max_length == 2000
    assert scorer.prompts_in_nli is True
    assert scorer.length_normalize is True


@pytest.mark.parametrize("scorer_name", SAMPLED_LOGPROBS_SCORER_NAMES)
def test_evaluate_with_mocked_scorers(scorer, scorer_name):
    """Test the evaluate method with mocked scorers."""
    responses = ["response1", "response2"]
    sampled_responses = [["sample1", "sample2"], ["sample3", "sample4"]]
    logprobs_results = [[{"logprob": -0.1}], [{"logprob": -0.2}]]
    sampled_logprobs_results = [[[{"logprob": -0.15}]], [[{"logprob": -0.25}]]]
    prompts = ["prompt1", "prompt2"]

    # Mock individual scorer methods
    with patch.object(scorer, "monte_carlo_probability", return_value=[0.5, 0.6]) as mock_mc, patch.object(scorer, "compute_consistency_confidence", return_value=[0.7, 0.8]) as mock_cc, patch.object(scorer, "compute_semantic_negentropy", return_value=[0.9, 1.0]) as mock_sn, patch.object(scorer, "compute_semantic_density", return_value=[1.1, 1.2]) as mock_sd:
        scorer.scorers = [scorer_name]
        result = scorer.evaluate(responses=responses, sampled_responses=sampled_responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results, prompts=prompts)

        # Verify the correct scorer method was called
        if scorer_name == "monte_carlo_probability":
            mock_mc.assert_called_once()
        elif scorer_name == "consistency_and_confidence":
            mock_cc.assert_called_once()
        elif scorer_name == "semantic_negentropy":
            mock_sn.assert_called_once()
        elif scorer_name == "semantic_density":
            mock_sd.assert_called_once()

        # Verify the result contains the correct scorer output
        assert scorer_name in result
        assert isinstance(result[scorer_name], list)


def test_monte_carlo_probability(scorer):
    """Test the monte_carlo_probability method."""
    responses = ["response1", "response2"]
    logprobs_results = [[{"logprob": -0.1}], [{"logprob": -0.2}]]
    sampled_logprobs_results = [[[{"logprob": -0.15}]], [[{"logprob": -0.25}]]]

    # Mock _compute_single_generation_scores
    with patch.object(scorer, "_compute_single_generation_scores", return_value=[0.8, 0.9]):
        result = scorer.monte_carlo_probability(responses=responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results)
        assert isinstance(result, list)
        assert len(result) == len(responses)


def test_monte_carlo_probability_excludes_logprob_sets_without_logprobs(scorer):
    """One logprob set missing its payload must not destroy the estimate from the rest."""
    responses = ["response1", "response2"]
    logprobs_results = [[{"logprob": -0.1}], [{"logprob": -0.2}]]
    sampled_logprobs_results = [[[{"logprob": -0.3}], None, [{"logprob": -0.5}]], [[{"logprob": -0.4}], [{"logprob": -0.6}]]]

    result = scorer.monte_carlo_probability(responses=responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results)

    assert not np.isnan(result[0])
    assert np.isclose(result[0], np.mean([np.exp(-0.1), np.exp(-0.3), np.exp(-0.5)]))
    assert np.isclose(result[1], np.mean([np.exp(-0.2), np.exp(-0.4), np.exp(-0.6)]))


def test_monte_carlo_probability_warns_when_logprob_sets_are_unusable(scorer):
    """The reduced effective sample count must be surfaced, not silent."""
    responses = ["response1", "response2"]
    logprobs_results = [[{"logprob": -0.1}], [{"logprob": -0.2}]]
    sampled_logprobs_results = [[[{"logprob": -0.3}], None, [{"logprob": -0.5}]], [[{"logprob": -0.4}], [{"logprob": -0.6}]]]

    with pytest.warns(UserWarning, match=r"Logprobs were unavailable for 1 of 7 logprob sets \(responses: \[0\]\)"):
        scorer.monte_carlo_probability(responses=responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results)


def test_monte_carlo_probability_returns_nan_when_all_sets_unusable(scorer):
    """A response with no usable logprobs stays NaN rather than getting a fabricated value."""
    with pytest.warns(UserWarning, match=r"Logprobs were unavailable for 2 of 2 logprob sets"):
        result = scorer.monte_carlo_probability(responses=["response1"], logprobs_results=[None], sampled_logprobs_results=[[None]])

    assert np.isnan(result[0])


def test_compute_consistency_confidence(scorer):
    """Test the compute_consistency_confidence method."""
    responses = ["response1", "response2"]
    sampled_responses = [["sample1", "sample2"], ["sample3", "sample4"]]
    logprobs_results = [[{"logprob": -0.1}], [{"logprob": -0.2}]]

    # Mock CosineScorer and _compute_single_generation_scores
    with patch("uqlm.black_box.cosine.CosineScorer.evaluate", return_value=[0.5, 0.6]), patch.object(scorer, "_compute_single_generation_scores", return_value=[0.7, 0.8]):
        result = scorer.compute_consistency_confidence(responses=responses, sampled_responses=sampled_responses, logprobs_results=logprobs_results)
        assert isinstance(result, list)
        assert len(result) == len(responses)


def test_compute_semantic_negentropy(scorer):
    """Test the compute_semantic_negentropy method."""
    responses = ["response1", "response2"]
    prompts = ["prompt1", "prompt2"]
    sampled_responses = [["sample1", "sample2"], ["sample3", "sample4"]]
    logprobs_results = [[{"logprob": -0.1}], [{"logprob": -0.2}]]
    sampled_logprobs_results = [[[{"logprob": -0.15}]], [[{"logprob": -0.25}]]]

    # Mock SemanticEntropy
    with patch("uqlm.scorers.shortform.entropy.SemanticEntropy.score", return_value=MagicMock(to_dict=lambda: {"data": {"tokenprob_confidence_scores": [0.9, 1.0]}})):
        result = scorer.compute_semantic_negentropy(responses=responses, prompts=prompts, sampled_responses=sampled_responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results)
        assert isinstance(result, list)
        assert len(result) == len(responses)


def test_compute_semantic_density(scorer):
    """Test the compute_semantic_density method."""
    responses = ["response1", "response2"]
    sampled_responses = [["sample1", "sample2"], ["sample3", "sample4"]]
    logprobs_results = [[{"logprob": -0.1}], [{"logprob": -0.2}]]
    sampled_logprobs_results = [[[{"logprob": -0.15}]], [[{"logprob": -0.25}]]]
    prompts = ["prompt1", "prompt2"]

    # Mock the semantic_negentropy_scorer and its clusterer
    mock_clusterer = MagicMock()
    mock_clusterer.nli.probabilities = [0.1, 0.2]
    mock_semantic_negentropy_scorer = MagicMock()
    mock_semantic_negentropy_scorer.clusterer = mock_clusterer

    # Assign the mocked semantic_negentropy_scorer to the scorer
    scorer.semantic_negentropy_scorer = mock_semantic_negentropy_scorer

    # Mock SemanticDensity
    with patch("uqlm.scorers.shortform.density.SemanticDensity.score", return_value=MagicMock(to_dict=lambda: {"data": {"semantic_density_values": [1.1, 1.2]}})):
        result = scorer.compute_semantic_density(responses=responses, sampled_responses=sampled_responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results, prompts=prompts)
        assert isinstance(result, list)
        assert len(result) == len(responses)
        assert result == [1.1, 1.2]


def test_heavyweight_members_reused_across_evaluate_calls(scorer):
    """Regression test: SemanticEntropy/SemanticDensity/CosineScorer were re-instantiated on every call."""
    mock_uq_result = MagicMock()
    mock_uq_result.to_dict.return_value = {"data": {"tokenprob_confidence_scores": [0.5], "semantic_density_values": [0.5]}}

    with patch("uqlm.white_box.sampled_logprobs.NLI") as mock_nli_class, patch("uqlm.white_box.sampled_logprobs.SemanticEntropy") as mock_se_class, patch("uqlm.white_box.sampled_logprobs.SemanticDensity") as mock_sd_class, patch("uqlm.white_box.sampled_logprobs.CosineScorer") as mock_cs_class:
        mock_se_class.return_value.score.return_value = mock_uq_result
        mock_sd_class.return_value.score.return_value = mock_uq_result
        mock_cs_class.return_value.evaluate.return_value = [0.5]

        kwargs = dict(responses=["r"], sampled_responses=[["s"]], logprobs_results=[[{"logprob": -0.1}]], sampled_logprobs_results=[[[{"logprob": -0.2}]]], prompts=["p"])
        for _ in range(2):
            scorer.compute_semantic_negentropy(**kwargs)
            scorer.compute_semantic_density(**kwargs)
        with patch.object(scorer, "_compute_single_generation_scores", return_value=[0.5]):
            for _ in range(2):
                scorer.compute_consistency_confidence(responses=["r"], sampled_responses=[["s"]], logprobs_results=[[{"logprob": -0.1}]])

    assert mock_nli_class.call_count == 1  # one shared NLI model
    assert mock_se_class.call_count == 1
    assert mock_sd_class.call_count == 1
    assert mock_cs_class.call_count == 1
    # SemanticEntropy and SemanticDensity must share the same NLI instance
    assert mock_se_class.call_args.kwargs["nli"] is mock_sd_class.call_args.kwargs["nli"]
