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

import torch


def get_best_device() -> torch.device:
    """
    Detects and returns the best available PyTorch device.
    Prioritizes CUDA (NVIDIA GPU), then MPS (macOS), then CPU.

    Returns
    -------
    torch.device
        The best available device.

    Examples
    --------
    >>> device = get_best_device()
    >>> print(f"Using {device.type} device")
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
