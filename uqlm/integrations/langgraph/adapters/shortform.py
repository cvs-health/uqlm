# Copyright 2026 CVS Health and/or one of its affiliates
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

from uqlm.scorers.shortform.black_box import BlackBoxUQ
from uqlm.scorers.shortform.white_box import WhiteBoxUQ
from uqlm.scorers.shortform.panel import LLMPanel
from uqlm.scorers.shortform.ensemble import UQEnsemble
from uqlm.scorers.shortform.entropy import SemanticEntropy
from uqlm.integrations.base import register_adapter

_SKIP_KEYS = frozenset({"prompts", "responses", "sampled_responses", "raw_responses", "raw_sampled_responses", "logprob", "sampled_logprob"})


def _extract_shortform_payload(result) -> dict:
    data = result.data
    responses_list = data.get("responses", [])
    scores = {k: v[0] for k, v in data.items() if k not in _SKIP_KEYS and isinstance(v, list) and len(v) > 0}
    primary = [responses_list[0]] if responses_list else []
    return {"scores": scores, "responses": primary, "extra": {}}


class BlackBoxUQAdapter:
    scorer_type = BlackBoxUQ

    async def run(self, scorer, *, prompt, response, mode, num_responses, sampled_responses=None, **kwargs):
        if mode == "score" and response is not None and sampled_responses is not None:
            result = scorer.score(responses=[response], sampled_responses=[sampled_responses], show_progress_bars=False)
        else:
            result = await scorer.generate_and_score(prompts=[prompt], num_responses=num_responses, show_progress_bars=False)
        return _extract_shortform_payload(result)


class WhiteBoxUQAdapter:
    scorer_type = WhiteBoxUQ

    async def run(self, scorer, *, prompt, response, mode, num_responses, logprobs_results=None, sampled_responses=None, sampled_logprobs_results=None, **kwargs):
        if mode == "score" and logprobs_results is not None:
            result = await scorer.score(
                logprobs_results=[logprobs_results],
                prompts=[prompt],
                responses=[response] if response is not None else None,
                sampled_responses=[sampled_responses] if sampled_responses is not None else None,
                sampled_logprobs_results=[sampled_logprobs_results] if sampled_logprobs_results is not None else None,
                show_progress_bars=False,
            )
        else:
            result = await scorer.generate_and_score(prompts=[prompt], num_responses=num_responses, show_progress_bars=False)
        return _extract_shortform_payload(result)


class LLMPanelAdapter:
    scorer_type = LLMPanel

    async def run(self, scorer, *, prompt, response, mode, num_responses, **kwargs):
        if mode == "score" and response is not None:
            result = await scorer.score(prompts=[prompt], responses=[response], show_progress_bars=False)
        else:
            result = await scorer.generate_and_score(prompts=[prompt], show_progress_bars=False)
        return _extract_shortform_payload(result)


class UQEnsembleAdapter:
    scorer_type = UQEnsemble

    async def run(self, scorer, *, prompt, response, mode, num_responses, sampled_responses=None, logprobs_results=None, **kwargs):
        if mode == "score" and response is not None:
            result = await scorer.score(
                prompts=[prompt],
                responses=[response],
                sampled_responses=[sampled_responses] if sampled_responses is not None else None,
                logprobs_results=[logprobs_results] if logprobs_results is not None else None,
                num_responses=num_responses,
                show_progress_bars=False,
            )
        else:
            result = await scorer.generate_and_score(prompts=[prompt], num_responses=num_responses, show_progress_bars=False)
        return _extract_shortform_payload(result)


class SemanticEntropyAdapter:
    scorer_type = SemanticEntropy

    async def run(self, scorer, *, prompt, response, mode, num_responses, sampled_responses=None, logprobs_results=None, sampled_logprobs_results=None, **kwargs):
        if mode == "score" and response is not None and sampled_responses is not None:
            result = scorer.score(
                prompts=[prompt],
                responses=[response],
                sampled_responses=[sampled_responses],
                logprobs_results=[logprobs_results] if logprobs_results is not None else None,
                sampled_logprobs_results=[sampled_logprobs_results] if sampled_logprobs_results is not None else None,
                show_progress_bars=False,
            )
        else:
            result = await scorer.generate_and_score(prompts=[prompt], num_responses=num_responses, show_progress_bars=False)
        return _extract_shortform_payload(result)


register_adapter(BlackBoxUQAdapter())
register_adapter(WhiteBoxUQAdapter())
register_adapter(LLMPanelAdapter())
register_adapter(UQEnsembleAdapter())
register_adapter(SemanticEntropyAdapter())
