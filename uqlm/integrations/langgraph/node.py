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

from uqlm.integrations.base import resolve_adapter
import uqlm.integrations.langgraph.adapters  # noqa: F401 — triggers adapter registration

# Allowed values for the ``mode`` argument. ``"score"`` scores precomputed
# responses already present in the state (falling back to generation when the
# required inputs are absent); ``"generate_and_score"`` always generates fresh
# responses with the scorer before scoring.
_VALID_MODES = ("score", "generate_and_score")

# State keys that the node forwards verbatim to the resolved adapter when
# present. These names intentionally match the uqlm scorer parameter names so
# the mapping stays one-to-one:
#   - ``sampled_responses``         -> scorer.score(sampled_responses=...)
#   - ``logprobs_results``          -> scorer.score(logprobs_results=...)
#   - ``sampled_logprobs_results``  -> scorer.score(sampled_logprobs_results=...)
_PASSTHROUGH_STATE_KEYS = ("sampled_responses", "logprobs_results", "sampled_logprobs_results")


class UQLMNode:
    """Wrap any uqlm scorer as an (async) LangGraph node.

    The node reads a single prompt (and optionally a single response plus
    precomputed scoring inputs) from the graph state, runs the scorer through
    its registered adapter, and writes the result back under ``output_key``.

    State keys read (names are configurable, defaults shown):
        prompt (``prompt``):
            Required. The prompt string for this item. Forwarded to the scorer
            as ``prompts=[prompt]``.
        response (``response``):
            Optional. A precomputed response string. Forwarded as
            ``responses=[response]`` when ``mode="score"``.
        sampled_responses:
            Optional ``List[str]`` of sampled responses for this prompt.
            Forwarded as ``sampled_responses=[sampled_responses]``.
        logprobs_results:
            Optional logprobs for the primary response (white-box scoring).
            Forwarded as ``logprobs_results=[logprobs_results]``.
        sampled_logprobs_results:
            Optional logprobs for the sampled responses. Forwarded as
            ``sampled_logprobs_results=[sampled_logprobs_results]``.

    State key written:
        output_key (``uq``): a dict payload with keys
            ``scores`` (dict of scorer-name -> score),
            ``responses`` (list with the primary response),
            ``extra`` (adapter-specific metadata), and
            ``scorer`` (the scorer class name).

    Args:
        scorer: A uqlm scorer instance (e.g. ``WhiteBoxUQ``, ``BlackBoxUQ``).
        output_key: State key to write the result payload under.
        mode: One of ``"score"`` or ``"generate_and_score"``. See ``_VALID_MODES``.
        prompt: State key to read the prompt from.
        response: State key to read the (optional) response from.
        num_responses: Number of responses to sample when generating.
        adapter_kwargs: Extra keyword arguments forwarded to the adapter.
    """

    def __init__(self, scorer, *, output_key: str = "uq", mode: str = "score", prompt: str = "prompt", response: str = "response", num_responses: int = 5, adapter_kwargs: dict | None = None):
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self.scorer = scorer
        self.output_key = output_key
        self.mode = mode
        self.prompt = prompt
        self.response = response
        self.num_responses = num_responses
        self.adapter_kwargs = adapter_kwargs

    async def __acall__(self, state: dict) -> dict:
        adapter = resolve_adapter(self.scorer)
        prompt = state[self.prompt]
        response = state.get(self.response)

        extra_kwargs = dict(self.adapter_kwargs or {})
        for key in _PASSTHROUGH_STATE_KEYS:
            if key in state and key not in extra_kwargs:
                extra_kwargs[key] = state[key]

        payload = await adapter.run(self.scorer, prompt=prompt, response=response, mode=self.mode, num_responses=self.num_responses, **extra_kwargs)
        payload["scorer"] = type(self.scorer).__name__
        return {self.output_key: payload}

    def __call__(self, state: dict) -> dict:
        return asyncio.run(self.__acall__(state))


def make_uqlm_node(scorer, **kwargs):
    """Return an async LangGraph node function wrapping ``scorer``.

    Convenience factory around :class:`UQLMNode`. The returned coroutine takes
    the graph state and returns the ``{output_key: payload}`` update, so it can
    be dropped directly into ``StateGraph.add_node``. All keyword arguments are
    forwarded to :class:`UQLMNode` (see its docstring for the accepted keys and
    the state contract).
    """
    node = UQLMNode(scorer, **kwargs)

    async def _node(state: dict) -> dict:
        return await node.__acall__(state)

    return _node
