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

import gc
import numpy as np
import pytest
from uqlm.nli.nli import NLI


@pytest.fixture
def text1():
    return "Question: What is captial of France, Answer: Paris"


@pytest.fixture
def text2():
    return "Question: What is captial of France, Answer: Capital of France is Paris city."


@pytest.fixture(scope="module")
def nli_model():
    return NLI(device="cpu")


@pytest.fixture
def nli_model_cpu():
    return NLI(verbose=True, device="cpu")


@pytest.mark.flaky(reruns=3)
def test_nli(text1, text2, nli_model):
    probabilities = nli_model.predict(text1, text2)
    assert abs(float(probabilities[0][0]) - 0.0012184) < 1e-5


# @pytest.mark.flaky(reruns=3)
# def test_nli2(text1, nli_model_cpu):
#     result = nli_model_cpu._observed_consistency_i(original=text1, candidates=[text1] * 5, use_best=False, compute_entropy=False)
#     assert result["nli_score_i"] == 1
#     assert result["discrete_semantic_entropy"] is None
#     assert result["tokenprob_semantic_entropy"] is None


@pytest.mark.flaky(reruns=3)
def test_nli3(text1, text2, nli_model_cpu):
    expected_warning = "Maximum response length exceeded for NLI comparison. Truncation will occur. To adjust, change the value of max_length"

    with pytest.warns(UserWarning, match=expected_warning):
        nli_model_cpu.predict(text1 * 50, text2)
    del nli_model_cpu
    gc.collect()


# @pytest.mark.flaky(reruns=3)
# def test_nli4(nli_model_cpu):
#     text1 = "Capital of France is Paris"
#     text2 = " Paris is the capital of France"
#     text3 = "Rome is the capital of Italy"
#     logprobs_results = [
#         [{"token": "Capital", "logprob": 0.6}, {"token": "of", "logprob": 0.5}, {"token": "France", "logprob": 0.3}, {"token": "is", "logprob": 0.3}, {"token": "Paris", "logprob": 0.3}],
#         [{"token": "Paris", "logprob": 0.75}, {"token": "is", "logprob": 0.8}, {"token": "the", "logprob": 0.9}, {"token": "capital", "logprob": 0.6}, {"token": "of", "logprob": 0.6}, {"token": "France", "logprob": 0.6}],
#         [{"token": "Rome", "logprob": 0.75}, {"token": "is", "logprob": 0.8}, {"token": "the", "logprob": 0.9}, {"token": "capital", "logprob": 0.6}, {"token": "of", "logprob": 0.6}, {"token": "Italy", "logprob": 0.6}],
#     ]
#     best_response, semantic_negentropy, nli_scores, tokenprob_semantic_entropy = nli_model_cpu._semantic_entropy_process(candidates=[text1, text2, text3], i=1, logprobs_results=logprobs_results)

#     assert best_response == text2
#     assert pytest.approx(semantic_negentropy, abs=1e-5) == 0.6365141682948128
#     assert pytest.approx(list(nli_scores.values()), abs=1e-5) == [0.9997053, 0.9997053, 0.24012965, 0.24012965]
#     assert pytest.approx(tokenprob_semantic_entropy, abs=1e-5) == 0.6918935849478249
#     del nli_model_cpu
#     gc.collect()


@pytest.mark.flaky(reruns=3)
def test_pair_encoding_is_canonical(nli_model, text1, text2):
    """Inputs must use the tokenizer's canonical pair encoding for the default model."""
    tokenizer = nli_model.tokenizer
    expected = tokenizer.build_inputs_with_special_tokens(tokenizer(text1, add_special_tokens=False).input_ids, tokenizer(text2, add_special_tokens=False).input_ids)
    assert tokenizer(text1, text2).input_ids == expected


@pytest.mark.flaky(reruns=3)
def test_predict_batch_matches_sequential(nli_model, text1, text2):
    """Batched inference must produce the same probabilities as one-pair-at-a-time inference."""
    pairs = [(text1, text2), (text2, text1), ("Today is a beautiful day", "It is quite rainy and unpleasant outside"), ("The sky is blue", "The sky is blue today"), (text1, text1)]
    batched = nli_model.predict_batch(pairs)
    sequential = np.concatenate([nli_model.predict(premise, hypothesis) for premise, hypothesis in pairs])
    assert batched.shape == (len(pairs), 3)
    assert np.allclose(batched, sequential, atol=1e-5)
    assert np.allclose(batched.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.flaky(reruns=3)
def test_predict_batch_chunking(nli_model, text1, text2):
    """Results must not depend on how pairs are split into forward-pass chunks."""
    pairs = [(text1, text2), (text2, text1), ("The sky is blue", "The ocean is deep"), ("Water is wet", "Fire is hot"), ("Cats are mammals", "Cats are animals")]
    original_batch_size = nli_model.batch_size
    try:
        nli_model.batch_size = 2
        chunked = nli_model.predict_batch(pairs)
    finally:
        nli_model.batch_size = original_batch_size
    unchunked = nli_model.predict_batch(pairs)
    assert np.allclose(chunked, unchunked, atol=1e-5)


@pytest.mark.flaky(reruns=3)
def test_predict_batch_empty(nli_model):
    result = nli_model.predict_batch([])
    assert result.shape == (0, 3)


@pytest.mark.flaky(reruns=3)
def test_cache_uses_tuple_keys_without_collision(nli_model):
    """Regression test: string cache keys f"{r1}_{r2}" collided for ("a_b", "c") vs ("a", "b_c")."""
    nli_model.probabilities = dict()
    result1 = nli_model.get_nli_results("a_b", "c")
    result2 = nli_model.get_nli_results("a", "b_c")
    assert set(nli_model.probabilities.keys()) == {("a_b", "c"), ("c", "a_b"), ("a", "b_c"), ("b_c", "a")}
    # Under the old underscore-joined keys, both pairs mapped to "a_b_c" and shared one entry
    assert not np.allclose(nli_model.probabilities[("a_b", "c")], nli_model.probabilities[("a", "b_c")])
    assert result1["noncontradiction_score"] != result2["noncontradiction_score"]


@pytest.mark.flaky(reruns=3)
def test_cache_hit_returns_stored_result(nli_model, text1, text2):
    """A cached pair must be reused without another forward pass."""
    nli_model.probabilities = dict()
    first = nli_model.get_nli_results(text1, text2)
    forward_calls = []
    original_predict_batch = nli_model.predict_batch

    def counting_predict_batch(pairs):
        forward_calls.append(pairs)
        return original_predict_batch(pairs)

    nli_model.predict_batch = counting_predict_batch
    try:
        second = nli_model.get_nli_results(text1, text2)
    finally:
        nli_model.predict_batch = original_predict_batch
    assert forward_calls == []
    assert first["noncontradiction_score"] == second["noncontradiction_score"]
    assert first["entailment"] == second["entailment"]


@pytest.mark.flaky(reruns=3)
def test_get_nli_results_batch_matches_single(nli_model, text1, text2):
    """Batch results must match single-pair results, including the identical-response shortcut."""
    nli_model.probabilities = dict()
    batch = nli_model.get_nli_results_batch([(text1, text2), (text1, text1), (text2, text1)])
    nli_model.probabilities = dict()
    singles = [nli_model.get_nli_results(text1, text2), nli_model.get_nli_results(text1, text1), nli_model.get_nli_results(text2, text1)]
    for batch_result, single_result in zip(batch, singles):
        assert batch_result["entailment"] == single_result["entailment"]
        assert abs(batch_result["noncontradiction_score"] - single_result["noncontradiction_score"]) < 1e-5
        assert abs(batch_result["entailment_score"] - single_result["entailment_score"]) < 1e-5
    assert batch[1] == {"noncontradiction_score": 1, "entailment": True, "entailment_score": 1}


@pytest.mark.flaky(reruns=3)
def test_token_level_truncation(nli_model):
    """Inputs beyond the model's token limit must be truncated instead of overflowing the model."""
    assert nli_model.max_tokens == 512
    long_text = "The quick brown fox jumps over the lazy dog and keeps running through the forest. " * 60
    with pytest.warns(UserWarning, match="Truncation will occur"):
        probabilities = nli_model.predict(long_text, long_text[:150])
    assert probabilities.shape == (1, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    encoded = nli_model.tokenizer(long_text[0 : nli_model.max_length], long_text[:150], truncation="longest_first", max_length=nli_model.max_tokens)
    assert len(encoded.input_ids) <= nli_model.max_tokens
