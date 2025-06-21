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


import os
import json
from dotenv import load_dotenv, find_dotenv

from uqlm.utils.dataloader import load_example_dataset
from uqlm.scorers import WhiteBoxUQ
from langchain_openai import AzureChatOpenAI


async def main():
    # svamp dataset to be used as a prod dataset
    svamp = load_example_dataset("svamp").rename(columns={"question_concat": "question", "Answer": "answer"})[["question", "answer"]].tail(5)

    # Define prompts
    MATH_INSTRUCTION = "When you solve this math problem only return the answer with no additional text.\n"
    prompts = [MATH_INSTRUCTION + prompt for prompt in svamp.question]

    # User to populate .env file with API credentials
    load_dotenv(find_dotenv())

    API_KEY = os.getenv("API_KEY")
    API_BASE = os.getenv("API_BASE")
    API_TYPE = os.getenv("API_TYPE")
    API_VERSION = os.getenv("API_VERSION")
    DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

    # This will be our main LLM for generation
    gpt = AzureChatOpenAI(
        deployment_name=DEPLOYMENT_NAME,
        openai_api_key=API_KEY,
        azure_endpoint=API_BASE,
        openai_api_type=API_TYPE,
        openai_api_version=API_VERSION,
        temperature=1,  # User to set temperature
    )

    wbuq = WhiteBoxUQ(llm=gpt)

    results = await wbuq.generate_and_score(prompts=prompts)

    results_file = "whitebox_results_file.json"
    with open(results_file, "w") as f:
        json.dump(results.to_dict(), f)


if __name__ == "__main__":
    main()
