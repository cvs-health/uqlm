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

import json
import os
import numpy as np
from uqlm.black_box import BertScorer, BLEURTScorer, CosineScorer, MatchScorer
import unittest

datafile_path = "tests/data/similarity/similarity_results_file.json"
with open(datafile_path, "r") as f:
    data = json.load(f)

responses = data["responses"]
sampled_responses = data["sampled_responses"]


def test_bert():
    bert = BertScorer()
    bert_result = bert.evaluate(responses=responses, sampled_responses=sampled_responses)
    assert all([abs(bert_result[i] - data["bert_result"][i]) < 1e-5 for i in range(len(bert_result))])


@unittest.skipIf((os.getenv("CI") == "true"), "Skipping test in CI due to dependency on GitHub repository.")
def test_bluert():
    bluert = BLEURTScorer()
    bluert_result = bluert.evaluate(responses=responses, sampled_responses=sampled_responses)
    assert all([abs(bluert_result[i] - data["bluert_result"][i]) < 1e-5 for i in range(len(bluert_result))])


def test_cosine(monkeypatch):
    embeddings1, embeddings2 = data["embeddings1"], data["embeddings2"]

    cosine = CosineScorer()

    # Mock return from  ('SentenceTransformer.encode' method)
    def mock_encode(*args, **kwargs):
        if len(embeddings1) >= len(embeddings2):
            return np.array(embeddings1.pop(0))
        return np.array(embeddings2.pop(0))

    monkeypatch.setattr(cosine.model, "encode", mock_encode)

    cosine_result = cosine.evaluate(responses=responses, sampled_responses=sampled_responses)
    assert all([abs(cosine_result[i] - data["cosine_result"][i]) < 1e-5 for i in range(len(cosine_result))])


def test_exact_match():
    match = MatchScorer()
    match_result = match.evaluate(responses=responses, sampled_responses=sampled_responses)
    assert all([abs(match_result[i] - data["match_result"][i]) < 1e-5 for i in range(len(match_result))])
