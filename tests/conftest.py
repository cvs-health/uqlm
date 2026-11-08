# Copyright 2025 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Disable MPS on macOS to prevent OOM failures during local test runs.
# (PyTorch will fall back to CPU.)
import functools

import torch


@functools.lru_cache
def _mps_disabled() -> bool:
    return False


# Keep the lru_cache wrapper: torch._dynamo reads `is_available.__wrapped__` at import.
if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = _mps_disabled
    torch.backends.mps.is_built = _mps_disabled
