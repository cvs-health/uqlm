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

from copy import deepcopy

import pytest

from uqlm.calibration.score_calibrator import ScoreCalibrator
from uqlm.utils.results import UQResult


@pytest.mark.parametrize("data", [{"prompts": ["q"], "responses": ["a"]}, {"responses": ["a"], "prompts": ["q"]}, {"responses": ["a"]}])
def test_result_exports_all_fields_without_modifying_input(data):
    payload = {"data": data, "metadata": {"model": "test"}, "parameters": {"temperature": 0}}
    original = deepcopy(payload)

    result = UQResult(payload)

    assert payload == original
    assert result.to_dict() == original
    if "prompts" in data:
        assert next(iter(result.data)) == "prompts"


@pytest.mark.parametrize("include_prompts", [False, True])
def test_added_scores_appear_in_both_exports(include_prompts):
    data = {"responses": ["a", "b"]}
    if include_prompts:
        data["prompts"] = ["q1", "q2"]
    result = UQResult({"data": data})

    result.data["calibrated_judge"] = [0.0, 1.0]

    assert result.to_dict()["data"]["calibrated_judge"] == [0.0, 1.0]
    assert result.to_df()["calibrated_judge"].tolist() == [0.0, 1.0]


def test_calibration_scores_remain_in_dictionary_export():
    result = UQResult({"data": {"prompts": ["q1", "q2"], "judge": [0.1, 0.9]}})

    ScoreCalibrator(method="isotonic").fit_transform(result, [0, 1])

    assert result.to_dict()["data"]["prompts"] == ["q1", "q2"]
    assert result.to_dict()["data"]["calibrated_judge"].tolist() == [0.0, 1.0]
