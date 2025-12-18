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

from typing import Dict, Any
import pandas as pd


class UQResult:
    def __init__(self, result: Dict[str, Any]) -> None:
        """
        Class that characterizes result of an UncertaintyQuantifier.

        Parameters
        ----------
        result: dict
            A dictionary that is defined during `evaluate` or `tune_params` method
        """
        data = result.get("data")
        if "prompts" in data:  # move prompts to front if exists
            prompts = data.pop("prompts")
            data = {"prompts": prompts, **data}

        self.data = data
        self.metadata = result.get("metadata")
        self.result_dict = result

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns result in dictionary form
        """
        return self.result_dict

    def to_df(self) -> pd.DataFrame:
        """
        Returns result in pd.DataFrame
        """
        rename_dict = {col: col[:-1] for col in self.data.keys() if col.endswith("s") and col not in ["sampled_responses", "raw_sampled_responses", "claims", "sentences"]}

        return pd.DataFrame(self.data).rename(columns=rename_dict)
