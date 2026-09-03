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

import os
import sys
import time

import pytest

from uqlm.code.code_evaluation import evaluate_python_code

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.name != "posix", reason="resource limits are enforced via POSIX rlimits")]


def _evaluate_single(code: str, expected_output: str, **kwargs) -> int:
    result = evaluate_python_code(responses=[code], public_test_cases=[[{"input": "", "output": expected_output}]], metadata=[{}], **kwargs)
    return result["unit_test_passed"][0]


def test_normal_grading_passes_under_default_limits():
    assert _evaluate_single("print(41 + 1)", "42") == 1


@pytest.mark.skipif(sys.platform != "linux", reason="macOS does not enforce RLIMIT_AS/RLIMIT_DATA")
def test_memory_limit_enforced():
    """An 8 GiB allocation must raise MemoryError inside the subprocess instead of exhausting host memory."""
    code = "try:\n    x = bytearray(8 * 1024**3)\n    print('ALLOCATED')\nexcept MemoryError:\n    print('CAPPED')"
    assert _evaluate_single(code, "CAPPED") == 1


def test_cpu_limit_kills_spinning_process():
    """A busy loop must be killed at the CPU limit well before the wall-clock ceiling."""
    code = "while True:\n    pass"
    start = time.monotonic()
    # timeout=15 keeps the per-test SIGALRM and the wall-clock kill far away, so
    # only the 2-second CPU rlimit can stop the loop this quickly.
    passed = _evaluate_single(code, "irrelevant", timeout=15, cpu_time_limit_seconds=2)
    elapsed = time.monotonic() - start
    assert passed == 0
    assert elapsed < 10, f"spinning process survived past the CPU limit ({elapsed:.1f}s)"


def test_file_size_limit_enforced(tmp_path):
    """Writing past the file-size cap must raise OSError inside the subprocess."""
    target = tmp_path / "big.bin"
    code = f"try:\n    with open({str(target)!r}, 'wb') as f:\n        f.write(b'x' * (5 * 1024 * 1024))\n    print('WROTE')\nexcept OSError:\n    print('CAPPED')"
    assert _evaluate_single(code, "CAPPED", max_file_size_bytes=1024**2) == 1
    assert not target.exists() or target.stat().st_size <= 1024**2
