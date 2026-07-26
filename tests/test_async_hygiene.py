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

import ast
import pathlib


def _base_names(cls: ast.ClassDef) -> set:
    """Collect the terminal names of a class's bases (e.g. unittest.TestCase -> TestCase)."""
    return {getattr(b, "attr", getattr(b, "id", "")) for b in cls.bases}


def test_no_async_tests_on_plain_testcase():
    """async def test_* methods on a plain unittest.TestCase are returned as un-awaited
    coroutines by the runner and never execute; such classes must derive from
    unittest.IsolatedAsyncioTestCase instead."""
    offenders = []
    for f in pathlib.Path(__file__).parent.rglob("test_*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            bases = _base_names(cls)
            if "TestCase" in bases and "IsolatedAsyncioTestCase" not in bases:
                for fn in cls.body:
                    if isinstance(fn, ast.AsyncFunctionDef) and fn.name.startswith("test"):
                        offenders.append(f"{f.name}::{cls.name}.{fn.name}")
    assert not offenders, f"async tests on plain unittest.TestCase do not execute; use IsolatedAsyncioTestCase: {offenders}"
