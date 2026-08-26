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

from uqlm.scorers.shortform.codegen import CodeGenUQ
from uqlm.integrations.base import register_adapter

_SKIP_KEYS = frozenset({"prompts", "responses", "sampled_responses", "raw_responses", "raw_sampled_responses", "logprob", "sampled_logprob"})


class CodeGenUQAdapter:
    scorer_type = CodeGenUQ

    async def run(self, scorer, *, prompt, response, mode, num_responses, sampled_responses=None, logprobs_results=None, sampled_logprobs_results=None, **kwargs):
        if mode == "score" and response is not None and sampled_responses is not None and logprobs_results is not None and sampled_logprobs_results is not None:
            result = await scorer.score(prompts=[prompt], responses=[response], sampled_responses=[sampled_responses], logprobs_results=[logprobs_results], sampled_logprobs_results=[sampled_logprobs_results], show_progress_bars=False)
        else:
            result = await scorer.generate_and_score(prompts=[prompt], num_responses=num_responses, show_progress_bars=False)
        data = result.data
        responses_list = data.get("responses", [])
        scores = {k: v[0] for k, v in data.items() if k not in _SKIP_KEYS and isinstance(v, list) and len(v) > 0}
        primary = [responses_list[0]] if responses_list else []
        return {"scores": scores, "responses": primary, "extra": {}}


register_adapter(CodeGenUQAdapter())
