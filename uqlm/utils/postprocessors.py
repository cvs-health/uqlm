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
