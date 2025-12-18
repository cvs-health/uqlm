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

from typing import List


# @misc{jiang2024graphbaseduncertaintymetricslongform,
#       title={Graph-based Uncertainty Metrics for Long-form Language Model Outputs},
#       author={Mingjian Jiang and Yangjun Ruan and Prasanna Sattigeri and Salim Roukos and Tatsunori Hashimoto},
#       year={2024},
#       eprint={2410.20783},
#       archivePrefix={arXiv},
#       primaryClass={cs.CL},
#       url={https://arxiv.org/abs/2410.20783},
# }
def get_response_reconstruction_prompt(claim_set: List[str]) -> str:
    """
    Parameters
    ----------
    claim_set: List[str]
        The response to be broken down into fact pieces.

    Returns
    -------
    str
        The prompt template for breaking down the response into fact pieces.
    """

    response_reconstruction_prompt = f"""
    Task: You are provided with a list of facts about prompt. Your
    goal is to synthesize these facts into a coherent paragraph. Use
    all the provided facts where possible, ensuring that no fact is
    misrepresented or overlooked. If there are redundant facts, choose
    the most comprehensive one for inclusion. The length of the paragraph
    should naturally reflect the number of provided facts—shorter for
    fewer facts and longer for more. Avoid unnecessary filler and focus
    on presenting the information clearly and concisely.
    The facts:
    {claim_set}
    """
    return response_reconstruction_prompt
