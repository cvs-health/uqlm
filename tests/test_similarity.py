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
import importlib.util
import sys
import shutil
import numpy as np
import subprocess
import importlib.resources as resources
import unittest
from uqlm.black_box import BertScorer, BLEURTScorer, CosineScorer, MatchScorer
from uqlm.black_box.baseclass.similarity_scorer import SimilarityScorer
from contextlib import redirect_stdout
import pytest
import io

datafile_path = "tests/data/similarity/similarity_results_file.json"
with open(datafile_path, "r") as f:
    data = json.load(f)

responses = data["responses"]
sampled_responses = data["sampled_responses"]


def test_bert():
    bert = BertScorer()
    bert_result = bert.evaluate(responses=responses, sampled_responses=sampled_responses)
    assert all([abs(bert_result[i] - data["bert_result"][i]) < 1e-5 for i in range(len(bert_result))])


def test_bluert_import_error():
    result = subprocess.run(["pip", "uninstall", "-y", "bleurt"], capture_output=True)
    result.check_returncode()  # Wait for subprocess to finish and check for errors
    bleurt_installed = importlib.util.find_spec("bleurt") is not None
    if not bleurt_installed:
        with pytest.raises(ImportError) as import_error:
            BLEURTScorer()
        assert "The bleurt package is required to use BLEURTScorer but is not installed. Please install it using:" in str(import_error.value)


@unittest.skipIf((os.getenv("CI") == "true"), "Skipping test in CI due to dependency on GitHub repository.")
def test_bluert_runtime_error(monkeypatch):
    resource_path = resources.files("uqlm.resources").joinpath("BLEURT-20")
    bluert_scorer_result = data["bluert_score"].copy()

    # Mock the entire bleurt module structure
    class MockBleurtScorer:
        def __init__(self, checkpoint):
            self.checkpoint = checkpoint
            if not os.listdir(resource_path):
                raise RuntimeError("Error Message")

        def score(self, references, candidates):
            return bluert_scorer_result.pop(0)

    # Create a proper module structure that matches the import path
    class MockScoreModule:
        BleurtScorer = MockBleurtScorer

    class MockBleurtModule:
        score = MockScoreModule()

    # Directly modify sys.modules dictionary with the complete module structure
    monkeypatch.setitem(sys.modules, "bleurt", MockBleurtModule())
    monkeypatch.setitem(sys.modules, "bleurt.score", MockScoreModule())

    shutil.rmtree(resource_path) if resource_path.is_dir() else None

    os.makedirs(resource_path, exist_ok=True)
    with pytest.raises(RuntimeError) as runtime_error:
        BLEURTScorer()
    assert "Failed to initialize BLEURT scorer. Error:" in str(runtime_error.value)
    shutil.rmtree(resource_path)

    bluert = BLEURTScorer()
    bluert_result = bluert.evaluate(responses=responses, sampled_responses=sampled_responses)
    assert all([abs(bluert_result[i] - data["bluert_result"][i]) < 1e-5 for i in range(len(bluert_result))])


def test_bleurt_unzip_print():
    test_path = resources.files("uqlm.resources")

    # Test corrupted zip file
    corrupted_zip_path = os.path.join(test_path, "corrupted.zip")
    with open(corrupted_zip_path, "w") as f:
        f.write("This is not a valid zip file")

    # Capture stdout
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        BLEURTScorer._unzip(corrupted_zip_path, test_path)

    # Check if anything was printed
    output = captured_output.getvalue()
    assert "Error: The downloaded BLEURT zip file is corrupted:" in output

    # Test general exception handling
    invalid_path = os.path.join(test_path, "nonexistent.zip")
    # Capture stdout
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        BLEURTScorer._unzip(invalid_path, test_path)  # Should handle exception without raising
    # Check if anything was printed
    output = captured_output.getvalue()
    assert "Unexpected error while extracting BLEURT zip file:" in output

    # Cleanup
    os.remove(corrupted_zip_path) if os.path.exists(corrupted_zip_path) else None


def test_bleurt_download_print():
    test_path = resources.files("uqlm.resources")
    # Use GitHub's 404 page which will reliably return a 404 status code
    test_url = "https://github.com/nonexistent-repo-that-will-never-exist-123456789/file.zip"

    # Capture stdout
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        BLEURTScorer._download(test_url, test_path)

    # Check if anything was printed
    output = captured_output.getvalue()
    assert "Failed to download file. Status code:" in output

    # Capture stdout
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        BLEURTScorer._download("http://invalid-url-that-doesnt-exist.com/file.zip", test_path)

    # Check if anything was printed
    output = captured_output.getvalue()
    assert "Network error occurred while downloading BLEURT checkpoint:" in output


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


def test_abstract_base_class():
    """Test to cover abstract base class"""

    class TestSimilarityScorer(SimilarityScorer):
        def __init__(self):
            super().__init__()

        def evaluate(self, responses, sampled_responses):
            super().evaluate(responses, sampled_responses)
            return [1.0]

    scorer = TestSimilarityScorer()
    result = scorer.evaluate(["test"], ["sample"])
    assert result == [1.0]
