import os, platform

# disable MPS on macOS to prevent all OOM failures.
import torch

if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False


# Automatically SKIP ALL transformer-heavy tests on Linux CI and Windows
def pytest_runtest_setup(item):
    system = platform.system()
    if system == "Windows" or (os.getenv("CI") == "true" and system == "Linux"):
        # Only skip the tests needing transformer models
        if "test_blackboxuq" in item.location[0] or "test_codegen" in item.location[0] or "test_similarity" in item.location[0] or "test_sampled_logprobs" in item.location[0] or "test_longtextuq" in item.location[0] or "test_longtextqa" in item.location[0]:
            import pytest

            pytest.skip("Skipping heavy transformer tests on Windows / Linux CI")
