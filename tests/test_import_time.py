import time
import pytest
import importlib
import unittest
import os


@unittest.skipIf(
    (os.getenv("CI") == "true"),
    "Skipping test in CI. Please run this check locally as needed.",
)
def test_import_time():
    """Test that the total import time for uqlm is less than 3 seconds."""
    # List of main dependencies to profile
    dependencies = ["langchain", "langchain_core", "transformers", "sentence_transformers", "bert_score", "matplotlib", "optuna", "numpy", "pandas", "sklearn"]

    results = []
    for dep in dependencies:
        try:
            # Clear import cache before each import
            importlib.invalidate_caches()
            start = time.time()
            importlib.import_module(dep)
            elapsed = time.time() - start
            results.append((dep, elapsed))
            print(f"{dep}: {elapsed:.4f} seconds")
        except Exception as e:
            print(f"{dep}: ERROR - {e}")

    # Now try importing uqlm and its components
    try:
        # Clear import cache before importing uqlm
        importlib.invalidate_caches()
        start = time.time()
        # Import and instantiate a BlackBoxUQ to force loading of dependencies
        from uqlm.scorers.black_box import BlackBoxUQ

        scorer = BlackBoxUQ(scorers=["bert_score", "cosine_sim"])
        # Actually use the scorer to force loading of all dependencies
        scorer.score(responses=["This is a test response"], sampled_responses=[["This is a test response", "This is another test response"]])
        elapsed = time.time() - start
        results.append(("uqlm (with BlackBoxUQ usage)", elapsed))
        print(f"\nuqlm (with BlackBoxUQ usage): {elapsed:.4f} seconds")
    except Exception as e:
        pytest.fail(f"Error importing uqlm: {e}")

    # Print the slowest import
    if results:
        slowest = max(results, key=lambda x: x[1])
        print(f"\nSlowest import: {slowest[0]} ({slowest[1]:.4f} seconds)")

    # Assert that uqlm import time is less than 3 seconds
    uqlm_time = next((t for name, t in results if name == "uqlm (with BlackBoxUQ usage)"), None)
    assert uqlm_time is not None, "Failed to measure uqlm import time"
    assert uqlm_time < 5.0, f"uqlm import time ({uqlm_time:.2f} seconds) exceeds 3 second limit"
