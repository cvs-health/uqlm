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
from uqlm.black_box.nli import NLIScorer


def test_nli():
    text1 = "Question: What is captial of France, Answer: Paris"
    text2 = "Question: What is captial of France, Answer: Capital of France is Paris city."

    nli_model = NLIScorer()
    probabilities = nli_model.predict(text1, text2)
    del nli_model
    gc.collect()
    assert abs(float(probabilities[0][0]) - 0.00159405) < 1e-5
