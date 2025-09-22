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
import pytest
from uqlm.black_box.nli import NLIScorer


@pytest.fixture
def text1():
    return "Question: What is captial of France, Answer: Paris"


@pytest.fixture
def text2():
    return "Question: What is captial of France, Answer: Capital of France is Paris city."


@pytest.fixture
def nli_model():
    return NLIScorer()


@pytest.fixture
def nli_model_cpu():
    return NLIScorer(verbose=True, device="cpu")


@pytest.mark.flaky(reruns=3)
def test_nli(text1, text2, nli_model):
    probabilities = nli_model.nli_model.predict(text1, text2)
    del nli_model
    gc.collect()
    assert abs(float(probabilities[0][0]) - 0.00159405) < 1e-5


@pytest.mark.flaky(reruns=3)
def test_nli2(text1, nli_model_cpu):
    result = nli_model_cpu._observed_consistency_i(original=text1, candidates=[text1] * 5, use_best=False, compute_entropy=False)
    assert result["nli_score_i"] == 1
    assert result["discrete_semantic_entropy"] is None
    assert result["tokenprob_semantic_entropy"] is None


@pytest.mark.flaky(reruns=3)
def test_nli3(text1, text2, nli_model_cpu):
    expected_warning = "Maximum response length exceeded for NLI comparison. Truncation will occur. To adjust, change the value of max_length"

    with pytest.warns(UserWarning, match=expected_warning):
        nli_model_cpu.nli_model.predict(text1 * 50, text2)
    del nli_model_cpu
    gc.collect()


@pytest.mark.flaky(reruns=3)
def test_nli4(nli_model_cpu):
    text1 = "Capital of France is Paris"
    text2 = " Paris is the capital of France"
    text3 = "Rome is the capital of Italy"
    logprobs_results = [
        [{"token": "Capital", "logprob": 0.6}, {"token": "of", "logprob": 0.5}, {"token": "France", "logprob": 0.3}, {"token": "is", "logprob": 0.3}, {"token": "Paris", "logprob": 0.3}],
        [{"token": "Paris", "logprob": 0.75}, {"token": "is", "logprob": 0.8}, {"token": "the", "logprob": 0.9}, {"token": "capital", "logprob": 0.6}, {"token": "of", "logprob": 0.6}, {"token": "France", "logprob": 0.6}],
        [{"token": "Rome", "logprob": 0.75}, {"token": "is", "logprob": 0.8}, {"token": "the", "logprob": 0.9}, {"token": "capital", "logprob": 0.6}, {"token": "of", "logprob": 0.6}, {"token": "Italy", "logprob": 0.6}],
    ]
    best_response, semantic_negentropy, nli_scores, tokenprob_semantic_entropy = nli_model_cpu._semantic_entropy_process(candidates=[text1, text2, text3], i=1, logprobs_results=logprobs_results)

    assert best_response == text2
    assert pytest.approx(semantic_negentropy, abs=1e-5) == 0.6365141682948128
    assert pytest.approx(list(nli_scores.values()), abs=1e-5) == [0.9997053, 0.9997053, 0.24012965, 0.24012965]
    assert pytest.approx(tokenprob_semantic_entropy, abs=1e-5) == 0.6918935849478249
    del nli_model_cpu
    gc.collect()
