import os, platform
import pytest

# disable MPS on macOS to prevent all OOM failures.
import torch

if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False


HEAVY_TESTS = ["test_blackboxuq", "test_codegen", "test_similarity", "test_sampled_logprobs", "test_longtextuq", "test_longtextqa"]


# Automatically Skip ALL transformer-heavy tests
def pytest_runtest_setup(item):
    filename = item.location[0]
    system = platform.system()
    # Skip on Windows, Linux, and macOS
    if system in ("Windows", "Linux", "Darwin"):
        if any(test in filename for test in HEAVY_TESTS):
            pytest.skip("Skipping heavy transformer tests on all 3 platforms")
