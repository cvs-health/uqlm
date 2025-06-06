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
import asyncio
import json
from dotenv import load_dotenv, find_dotenv
from uqlm.judges import LLMJudge
from uqlm.utils import ResponseGenerator
from langchain_openai import AzureChatOpenAI


async def main():
    # This notebook generate results based on these input & using "exai-gpt-35-turbo-16k" model
    prompts = [
        "Which part of the human body produces insulin?",
        "What color are the two stars on the national flag of Syria",
        "How many 'm's are there in the word strawberry",
    ]

    # User to populate .env file with API credentials
    load_dotenv(find_dotenv())

    API_KEY = os.getenv("API_KEY")
    API_BASE = os.getenv("API_BASE")
    API_TYPE = os.getenv("API_TYPE")
    API_VERSION = os.getenv("API_VERSION")
    DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

    original_llm = AzureChatOpenAI(
        deployment_name=DEPLOYMENT_NAME,
        openai_api_key=API_KEY,
        azure_endpoint=API_BASE,
        openai_api_type=API_TYPE,
        openai_api_version=API_VERSION,
        temperature=1,  # User to set temperature
    )
    
    rg = ResponseGenerator(llm=original_llm, max_calls_per_min=250)
    generations = await rg.generate_responses(prompts=prompts, count=1)
    responses = generations["data"]["response"]

    judge = LLMJudge(llm=original_llm, max_calls_per_min=250)

    judge_result = await judge.judge_responses(prompts=prompts, responses=responses)

    extract_answer = judge._extract_answers(responses=judge_result["judge_responses"])
    
    # Generate data for all templates
    templates = ["true_false_uncertain", "true_false", "continuous", "likert"]
    # Structure: one file with all template data
    all_results = {
       "prompts": prompts,
       "responses": responses,
       "templates": {}  # This will hold data for each template
    }
    for template in templates:
       judge = LLMJudge(llm=original_llm, max_calls_per_min=250, scoring_template=template)
       judge_result = await judge.judge_responses(prompts=prompts, responses=responses)
       extract_answer = judge._extract_answers(responses=judge_result["judge_responses"])
       # Store results for this template
       all_results["templates"][template] = {
           "judge_result": judge_result,
           "extract_answer": extract_answer,
       }
    # Save single comprehensive file
    results_file = "llmjudge_results_file.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f)
   
if __name__ == '__main__':
   asyncio.run(main())    