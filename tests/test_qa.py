import pytest
from langchain_openai import AzureChatOpenAI
from uqlm.scorers.longform.qa import LongTextQA

@pytest.fixture
def mock_llm():
    """Define mock LLM object using pytest.fixture."""
    return AzureChatOpenAI(deployment_name="YOUR-DEPLOYMENT", temperature=1, api_key="SECRET_API_KEY", api_version="2024-05-01-preview", azure_endpoint="https://mocked.endpoint.com")

def test_init_with_defaults(mock_llm):
    qa = LongTextQA(llm=mock_llm)
    assert qa.granularity == "claim"
    assert qa.aggregation == "mean"
    assert qa.response_refinement is False
    assert qa.system_prompt == "You are a helpful assistant."

def test_init_with_all_args(mock_llm, mock_claim_decomposition_llm, mock_question_generator_llm):
    scorers = ["entailment", "bert_score"]
    qa = LongTextQA(
        llm=mock_llm,
        scorers=scorers,
        granularity="sentence",
        aggregation="min",
        response_refinement=True,
        claim_filtering_scorer="entailment",
        system_prompt="System prompt here.",
        claim_decomposition_llm=mock_claim_decomposition_llm,
        question_generator_llm=mock_question_generator_llm,
        sampling_temperature=0.8,
        max_calls_per_min=10,
        questioner_max_calls_per_min=5,
        max_length=222,
        device="cpu",
        use_n_param=True,
    )
    assert qa.scorers == scorers
    assert qa.granularity == "sentence"
    assert qa.aggregation == "min"
    assert qa.response_refinement is True
    assert qa.claim_filtering_scorer == "entailment"
    assert qa.system_prompt == "System prompt here."
    assert qa.claim_decomposition_llm == mock_claim_decomposition_llm
    assert qa.question_generator_llm == mock_question_generator_llm
    assert qa.sampling_temperature == 0.8
    assert qa.max_calls_per_min == 10
    assert qa.questioner_max_calls_per_min == 5
    assert qa.max_length == 222
    assert qa.device == "cpu"
    assert qa.use_n_param is True

def test_claim_decomposition_llm_fallback(mock_llm):
    # If claim_decomposition_llm is None, should fallback to llm (property set)
    qa = LongTextQA(llm=mock_llm, claim_decomposition_llm=None)
    assert hasattr(qa, "claim_decomposition_llm")

def test_question_generator_llm_set(mock_llm):
    qa = LongTextQA(llm=mock_llm, question_generator_llm=mock_llm)
    assert qa.question_generator_llm == mock_llm
