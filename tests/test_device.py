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

from unittest.mock import patch

import pytest
import torch

from uqlm.utils.device import get_best_device


def test_get_best_device_returns_cpu_when_no_gpu():
    with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=False):
        device = get_best_device()
    assert device == torch.device("cpu")


def test_get_best_device_returns_cuda_when_available():
    with patch("torch.cuda.is_available", return_value=True):
        device = get_best_device()
    assert device == torch.device("cuda")


def test_get_best_device_returns_mps_when_cuda_unavailable_and_mps_available():
    with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=True):
        device = get_best_device()
    assert device == torch.device("mps")
