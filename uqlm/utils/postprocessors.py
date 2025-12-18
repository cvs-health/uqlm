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

from typing import Any, Dict, List


def claims_dicts_to_lists(claims_data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Extract claim-level data into list of lists"""
    return_dict = {k: [] for k in claims_data[0][0].keys()}
    for key in return_dict:
        claims_data_lists = []
        for i in range(len(claims_data)):
            claims_data_i = []
            for j in range(len(claims_data[i])):
                claims_data_i.append(claims_data[i][j][key])
            claims_data_lists.append(claims_data_i)
        return_dict[key] = claims_data_lists
    return return_dict


def math_postprocessor(input_string: str) -> str:
    """
    Parameters
    ----------

    input_string: str
        The string from which the numerical answer will be extracted. Only the integer part is extracted.

    Returns
    -------
    str
        The postprocessed string containing the integer part of the answer.
    """
    result = ""
    for char in input_string:
        if char.isdigit():
            result += char
        elif char == ".":
            break
    return result
