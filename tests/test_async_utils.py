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

import asyncio
import threading

import pytest

from uqlm.utils.async_utils import run_sync


async def _add(a, b, delay=0.0):
    if delay:
        await asyncio.sleep(delay)
    return a + b


async def _boom():
    await asyncio.sleep(0)
    raise ValueError("boom")


async def _current_thread_name():
    await asyncio.sleep(0)
    return threading.current_thread().name


def test_run_sync_without_running_loop():
    """When called from a thread with no running event loop, run_sync should drive the coroutine
    to completion on the calling thread (via asyncio.run) and return its result."""
    result = run_sync(_add(2, 3))
    assert result == 5


def test_run_sync_without_running_loop_runs_on_calling_thread():
    """With no running loop, run_sync should not need to spawn a worker thread."""
    name = run_sync(_current_thread_name())
    assert name == threading.current_thread().name


def test_run_sync_propagates_exceptions_without_running_loop():
    with pytest.raises(ValueError, match="boom"):
        run_sync(_boom())


@pytest.mark.asyncio
async def test_run_sync_inside_running_loop():
    """This is the scenario that matters for notebook users: the calling thread already has an
    active event loop (pytest-asyncio provides one here, mirroring what Jupyter/IPython does), so
    a plain `asyncio.run(coro)` would raise `RuntimeError: asyncio.run() cannot be called from a
    running event loop`. run_sync must instead complete the coroutine anyway."""
    result = run_sync(_add(2, 3))
    assert result == 5


@pytest.mark.asyncio
async def test_run_sync_inside_running_loop_uses_worker_thread():
    """Verifies run_sync actually takes the worker-thread fallback path (rather than, say, silently
    reusing the caller's loop in a way that would deadlock) when a loop is already running."""
    name = run_sync(_current_thread_name())
    assert name != threading.current_thread().name


@pytest.mark.asyncio
async def test_run_sync_propagates_exceptions_inside_running_loop():
    with pytest.raises(ValueError, match="boom"):
        run_sync(_boom())
