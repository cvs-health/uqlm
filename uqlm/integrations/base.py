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

from typing import Protocol, runtime_checkable


@runtime_checkable
class ScorerAdapter(Protocol):
    scorer_type: type

    async def run(self, scorer, *, prompt: str, response: str | None, mode: str, num_responses: int, **kwargs) -> dict: ...


_REGISTRY: list = []


def register_adapter(adapter) -> object:
    _REGISTRY.append(adapter)
    return adapter


def resolve_adapter(scorer) -> object:
    for a in _REGISTRY:
        if isinstance(scorer, a.scorer_type):
            return a
    raise TypeError(f"No UQLM adapter registered for {type(scorer).__name__}")
