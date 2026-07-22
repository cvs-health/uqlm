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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run a coroutine to completion from synchronous code and return its result.

    This is used to offer blocking, non-async counterparts (e.g. `generate_and_score_sync`)
    to methods that are otherwise only available as `async def` coroutines, so that users
    who are not working within an event loop are not required to write `async`/`await`
    boilerplate themselves.

    If the calling thread has no running event loop, the coroutine is executed directly
    with `asyncio.run`. If the calling thread already has a running event loop--for example,
    Jupyter/IPython kernels run a persistent event loop in the main thread--`asyncio.run`
    cannot be used directly since it raises `RuntimeError: asyncio.run() cannot be called
    from a running event loop`. In that case, the coroutine is instead executed to completion
    on a fresh event loop in a dedicated worker thread so it does not conflict with the
    caller's running loop.

    Parameters
    ----------
    coro : Coroutine
        The coroutine to execute (e.g. the object returned by calling an `async def` method,
        prior to awaiting it).

    Returns
    -------
    Any
        The value returned upon completion of `coro`.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No event loop is running in this thread, so it is safe to drive the
        # coroutine with a loop of our own via asyncio.run.
        return asyncio.run(coro)

    # A loop is already running in this thread. Hand the coroutine to a separate
    # worker thread, which runs it on its own event loop via asyncio.run, so we
    # do not attempt to nest event loops in the calling thread.
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()
