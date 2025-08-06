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

from uqlm.utils.prompt_templates import get_claim_breakdown_template
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, Console
from langchain_core.language_models.chat_models import BaseChatModel
import re


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


def claims_postprocessor(llm: BaseChatModel, responses: list[str] | str, progress: bool = True) -> list[dict]:
        """
        Parameters
        ----------
        responses: list[str] | str
            LLM response that will be decomposed into independent claims.
        progress: bool
            Whether to show a progress bar.
        """
        if isinstance(responses, str):
            responses = [responses]
        if progress:
            with Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeElapsedColumn(),
                console=Console()
            ) as progress:
                claims = []
                task = progress.add_task("Processing", total=len(responses))
                for response in responses:
                    progress.update(task, description="Processing response(s)...")
                    res = llm.invoke(get_claim_breakdown_template(response))
                    if res.content:
                        claims.append(re.findall(r"### (.*)", res.content))
                    progress.update(task, advance=1)
        else:
            claims = []
            for response in responses:
                res = llm.invoke(get_claim_breakdown_template(response))
                if res.content:
                    claims.append(re.findall(r"### (.*)", res.content))
        return claims



# def claims_postprocessor(input_string: str) -> str:
#     """
#     Parameters
#     ----------

#     input_string: str
#         LLM response that will be decomposed into independent claims.
#     """

#     df_subset = df.iloc[:2]
#     max_entity_len = df_subset["entity"].str.len().max()+2
#     responses = []
#     task = progress.add_task("[cyan]Processing", total=len(df_subset))
#     for i,row in df_subset.iterrows():
#         progress.update(task, description=f"Generating llm response for entity: '{row["entity"]:<{max_entity_len}}'")
#         prompt = row["factscore_prompt"]
#         response = test_llm.invoke(prompt)
#         responses.append(response.content)
#         current_progress = progress.tasks[task].completed
#         total_progress = progress.tasks[task].total
#         print(f"Debug: Iteration {i+1}, Progress: {current_progress}/{total_progress}")
#         progress.update(task, advance=1)
