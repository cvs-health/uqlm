import pytest
import math
from unittest.mock import patch, MagicMock

from uqlm.code.codebleu import CodeBLEU



# Fixture: mock external codebleu package

@pytest.fixture
def mock_codebleu_module():
    """Mock the external `codebleu` module."""
    mock_module = MagicMock()
    mock_module.calc_codebleu = MagicMock(return_value={"codebleu": 0.75})
    return mock_module


# Constructor tests

def test_constructor_importerror_when_codebleu_missing():
    """Should raise error when codebleu is not installed."""
    with patch("importlib.util.find_spec", return_value=None):
        with pytest.raises(ImportError):
            CodeBLEU()


def test_constructor_succeeds_when_codebleu_present(mock_codebleu_module):
    """Constructor succeeds and stores calc_codebleu."""
    with patch("importlib.util.find_spec", return_value=True), \
         patch.dict("sys.modules", {"codebleu": mock_codebleu_module}):

        cb = CodeBLEU()
        assert cb.calc_codebleu is mock_codebleu_module.calc_codebleu



# evaluate() tests

def test_evaluate_basic(mock_codebleu_module):
    """Evaluate should compute scores for each prompt."""
    with patch("importlib.util.find_spec", return_value=True), \
         patch.dict("sys.modules", {"codebleu": mock_codebleu_module}):

        cb = CodeBLEU()

        responses = ["print(1)", "x=2"]
        sampled = [["print(1)"], ["x=2"]]

        scores = cb.evaluate(responses, sampled)

        assert scores == [0.75, 0.75]
        assert cb.pair_scores[0] == [0.75]
        assert cb.pair_scores[1] == [0.75]


def test_evaluate_raises_on_empty_inputs(mock_codebleu_module):
    """evaluate() must reject empty inputs."""
    with patch("importlib.util.find_spec", return_value=True), \
         patch.dict("sys.modules", {"codebleu": mock_codebleu_module}):

        cb = CodeBLEU()

        with pytest.raises(ValueError):
            cb.evaluate([], [["x"]])

        with pytest.raises(ValueError):
            cb.evaluate(["x"], [])



# codebleu_confidence() tests

def test_codebleu_confidence_average(mock_codebleu_module):
    """Should return averaged confidence + pair scores."""
    with patch("importlib.util.find_spec", return_value=True), \
         patch.dict("sys.modules", {"codebleu": mock_codebleu_module}):

        cb = CodeBLEU()

        score, pairs = cb.codebleu_confidence("a", ["b", "c"])
        assert score == 0.75
        assert pairs == [0.75, 0.75]


def test_codebleu_confidence_empty_samples(mock_codebleu_module):
    """Empty samples → CodeBLEU returns ONLY float(nan)."""
    with patch("importlib.util.find_spec", return_value=True), \
         patch.dict("sys.modules", {"codebleu": mock_codebleu_module}):

        cb = CodeBLEU()

        result = cb.codebleu_confidence("a", [])
        assert isinstance(result, float)
        assert math.isnan(result)


def test_codebleu_confidence_all_nan(mock_codebleu_module):
    """If all pair scores are NaN, confidence must be NaN."""
    mock_codebleu_module.calc_codebleu = MagicMock(
        return_value={"codebleu": float("nan")}
    )

    with patch("importlib.util.find_spec", return_value=True), \
         patch.dict("sys.modules", {"codebleu": mock_codebleu_module}):

        cb = CodeBLEU()

        score, pairs = cb.codebleu_confidence("a", ["b"])
        assert math.isnan(score)
        assert math.isnan(pairs[0])



# codebleu_pair() tests

def test_codebleu_pair_basic(mock_codebleu_module):
    """Simple pair scoring returns mocked 0.75."""
    with patch("importlib.util.find_spec", return_value=True), \
         patch.dict("sys.modules", {"codebleu": mock_codebleu_module}):

        cb = CodeBLEU()
        score = cb.codebleu_pair("a", "b")
        assert score == 0.75


def test_codebleu_pair_empty(mock_codebleu_module):
    """Empty strings → return NaN."""
    with patch("importlib.util.find_spec", return_value=True), \
         patch.dict("sys.modules", {"codebleu": mock_codebleu_module}):

        cb = CodeBLEU()

        assert math.isnan(cb.codebleu_pair("", "b"))
        assert math.isnan(cb.codebleu_pair("a", ""))


def test_codebleu_pair_exception_returns_nan(mock_codebleu_module):
    """Exceptions in calc_codebleu should return NaN."""
    mock_codebleu_module.calc_codebleu = MagicMock(side_effect=Exception("fail"))

    with patch("importlib.util.find_spec", return_value=True), \
         patch.dict("sys.modules", {"codebleu": mock_codebleu_module}):

        cb = CodeBLEU()
        score = cb.codebleu_pair("a", "b")
        assert math.isnan(score)