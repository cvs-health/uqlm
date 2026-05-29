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

from uqlm.scorers.longform.longtext import LongTextUQ
from uqlm.scorers.longform.qa import LongTextQA
from uqlm.scorers.longform.graph import LongTextGraph
from uqlm.integrations.base import register_adapter

_SKIP_KEYS = frozenset({"prompts", "responses", "sampled_responses", "raw_responses", "raw_sampled_responses", "logprob", "sampled_logprob"})
_EXTRA_KEYS = frozenset({"claims_data", "refined_response"})


def _extract_longform_payload(result) -> dict:
    data = result.data
    responses_list = data.get("responses", [])
    scores = {}
    extra = {}
    for k, v in data.items():
        if k in _SKIP_KEYS:
            continue
        if k in _EXTRA_KEYS:
            extra[k] = v[0] if isinstance(v, list) and len(v) > 0 else v
        elif isinstance(v, list) and len(v) > 0:
            scores[k] = v[0]
    primary = [responses_list[0]] if responses_list else []
    return {"scores": scores, "responses": primary, "extra": extra}


class LongTextUQAdapter:
    scorer_type = LongTextUQ

    async def run(self, scorer, *, prompt, response, mode, num_responses, sampled_responses=None, **kwargs):
        if mode == "score" and response is not None and sampled_responses is not None:
            result = await scorer.score(responses=[response], sampled_responses=[sampled_responses], show_progress_bars=False)
        else:
            result = await scorer.generate_and_score(prompts=[prompt], num_responses=num_responses, show_progress_bars=False)
        return _extract_longform_payload(result)


class LongTextQAAdapter:
    scorer_type = LongTextQA

    async def run(self, scorer, *, prompt, response, mode, num_responses, **kwargs):
        if mode == "score" and response is not None:
            result = await scorer.score(prompts=[prompt], responses=[response], show_progress_bars=False)
        else:
            result = await scorer.generate_and_score(prompts=[prompt], show_progress_bars=False)
        return _extract_longform_payload(result)


class LongTextGraphAdapter:
    scorer_type = LongTextGraph

    async def run(self, scorer, *, prompt, response, mode, num_responses, sampled_responses=None, **kwargs):
        if mode == "score" and response is not None and sampled_responses is not None:
            result = await scorer.score(responses=[response], sampled_responses=[sampled_responses], show_progress_bars=False)
        else:
            result = await scorer.generate_and_score(prompts=[prompt], num_responses=num_responses, show_progress_bars=False)
        return _extract_longform_payload(result)


register_adapter(LongTextUQAdapter())
register_adapter(LongTextQAAdapter())
register_adapter(LongTextGraphAdapter())
