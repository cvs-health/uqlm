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


def test_nli(text1, text2, nli_model):
    probabilities = nli_model.predict(text1, text2)
    del nli_model
    gc.collect()
    assert abs(float(probabilities[0][0]) - 0.00159405) < 1e-5


def test_nli2(text1, nli_model_cpu):
    result = nli_model_cpu._observed_consistency_i(original=text1, candidates=[text1] * 5, use_best=False, compute_entropy=False)
    assert result["nli_score_i"] == 1
    assert result["semantic_negentropy"] is None


def test_nli3(text1, text2, nli_model_cpu):
    expected_warning = "Maximum response length exceeded for NLI comparison. Truncation will occur. To adjust, change the value of max_length"

    with pytest.warns(UserWarning, match=expected_warning):
        nli_model_cpu.predict(text1 * 50, text2)
    del nli_model_cpu
    gc.collect()


def test_nli4(text1, nli_model_cpu):
    with pytest.raises(ValueError) as value_error:
        nli_model_cpu._semantic_entropy_process(candidates=[text1] * 5, i=1, discrete=False)
    assert "SemanticEntropy currently only supports discrete evaluations" == str(value_error.value)

    del nli_model_cpu
    gc.collect()
