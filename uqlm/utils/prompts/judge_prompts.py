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

"""
Prompt templates for LLM-as-a-Judge scorers.

This module contains all prompt templates used by the LLM judge system.
"""

# =============================================================================
# DEFINITIONS
# =============================================================================

# Scoring configuration for all template types
SCORING_CONFIG = {
    "continuous": {"range": "0 (lowest) and 100 (highest)", "score_format": "[0-100]", "description": "Continuous scoring from 0-100"},
    "likert": {
        "range": "1 to 5, with 5 being the highest",
        "score_format": "[1-5]",
        "description": "Likert scale scoring from 1-5",
        "scale_definitions": """1 - Completely incorrect: The answer is entirely wrong or irrelevant.
                                2 - Mostly incorrect: The answer contains significant errors or misconceptions.
                                3 - Partially correct: The answer has some correct elements but also contains errors.
                                4 - Mostly correct: The answer is largely accurate with only minor errors or omissions.
                                5 - Completely correct: The answer is fully accurate and comprehensive.""",
    },
    "categorical": {"choices_2_class": '"Correct", "Incorrect"', "choices_3_class": '"Correct", "Incorrect", or "I am not sure"', "score_format": "{choices}", "description": "Categorical scoring with predefined choices"},
}

# Common instruction strings
COMMON_INSTRUCTIONS = {
    "only_choices": "YOUR ANSWER MUST ONLY CONTAIN ONE OF {choices}",
    "do_not_answer": "DO NOT ANSWER THE QUESTION AGAIN",
    "determine_correctness": "ONLY DETERMINE IF THE ANSWER TO THE QUESTION IS {choices}",
    "follow_format": "YOUR ANSWER MUST FOLLOW THIS EXACT FORMAT",
    "only_numerical": "ONLY RETURN YOUR NUMERICAL SCORE WITH NO SURROUNDING TEXT OR EXPLANATION",
    "confidence_rating": "THE CONFIDENCE RATING YOU PROVIDE MUST BE BETWEEN 0 and 100",
}

# =============================================================================
# EXAMPLES FOR PROMPTS
# =============================================================================


# Standardized example format (consistent across all templates)
def create_example(example_num: int, question: str, proposed_answer: str, response: str) -> str:
    """Create a standardized example format."""
    return f"""# Example {example_num}
                ## Data to analyze
                Question: {question}
                Proposed Answer: {proposed_answer}

                ## Your response
                {response}"""


# Common examples using standardized format
BENJAMIN_FRANKLIN_EXAMPLE = create_example(1, "Who was the first president of the United States?", "Benjamin Franklin", "4 (highly certain the proposed answer is incorrect)")

ARITHMETIC_EXAMPLE = create_example(2, "What is 2+2?", "4", "99 (highly certain the proposed answer is correct)")

# Explanation examples using standardized format
BENJAMIN_FRANKLIN_CATEGORICAL_EXPLANATION_EXAMPLE = create_example(
    1,
    "Who was the first president of the United States?",
    "Benjamin Franklin",
    """Score: Incorrect
                Explanation: The answer mentions Benjamin Franklin, but George Washington was the first president.""",
)

BENJAMIN_FRANKLIN_LIKERT_EXPLANATION_EXAMPLE = create_example(
    1,
    "Who was the first president of the United States?",
    "Benjamin Franklin",
    """Score: 1
                Explanation: The answer mentions Benjamin Franklin, but George Washington was the first president.""",
)

BENJAMIN_FRANKLIN_CONTINUOUS_EXPLANATION_EXAMPLE = create_example(
    1,
    "Who was the first president of the United States?",
    "Benjamin Franklin",
    """"Score: 4
                Explanation: Benjamin Franklin was never president. George Washington was the first president of the United States.""",
)

ARITHMETIC_EXPLANATIONS_EXAMPLE = create_example(
    2,
    "What is 2+2?",
    "4",
    """Score: 99
                Explanation: This is a basic arithmetic question and 2+2=4 is correct.""",
)

# =============================================================================
# PROMPT TEMPLATE DATABENJAMIN_FRANKLIN_LIKERT_EXPLANATION_EXAMPLE
# =============================================================================

PROMPT_TEMPLATES = {
    "continuous": {
        "base": f"""How likely is the above answer to be correct? 
                Analyze the answer and give your confidence in this answer between {SCORING_CONFIG["continuous"]["range"]}, 
                with 100 being certain the answer is correct, and 0 being certain the answer is incorrect.
                """,
        "score_format": f"Score: {SCORING_CONFIG['continuous']['score_format']}",
        "examples_with": f"""{BENJAMIN_FRANKLIN_CONTINUOUS_EXPLANATION_EXAMPLE}\n
                {ARITHMETIC_EXPLANATIONS_EXAMPLE}""",
        "examples_without": f"""{BENJAMIN_FRANKLIN_EXAMPLE}\n
                {ARITHMETIC_EXAMPLE}""",
        "instruction_without_explanations": f"""{COMMON_INSTRUCTIONS["confidence_rating"]}. {COMMON_INSTRUCTIONS["only_numerical"]}""",
        "instruction_with_explanations": f"""{COMMON_INSTRUCTIONS["confidence_rating"]}.""",
    },
    "likert": {
        "base": f"""You are a fair assessment expert evaluating the CORRECTNESS of an answer to a question.
                    Your task is to score the answer on a scale from {SCORING_CONFIG["likert"]["range"]}:
                    {SCORING_CONFIG["likert"]["scale_definitions"]}
                """,
        "score_format": f"Score: {SCORING_CONFIG['likert']['score_format']}",
        "examples_with": f"{BENJAMIN_FRANKLIN_LIKERT_EXPLANATION_EXAMPLE}",
        "examples_without": "",
        "instruction_without_explanations": f"Give only the numerical score ({SCORING_CONFIG['likert']['score_format']}).",
        "instruction_with_explanations": "Provide both score and explanation in the specified format.",
    },
    "categorical": {
        "base": """Your task is to look at the question and answer provided and determine if the answer is correct.""",
        "score_format": f"Score: {SCORING_CONFIG['categorical']['score_format']}",
        "examples_with": f"{BENJAMIN_FRANKLIN_CATEGORICAL_EXPLANATION_EXAMPLE}",
        "examples_without": "",
        "instruction_without_explanations": f"""You are to respond with ONLY one of: {{choices}}. 
                                {COMMON_INSTRUCTIONS["only_choices"]}. 
                                {COMMON_INSTRUCTIONS["do_not_answer"]}. 
                                {COMMON_INSTRUCTIONS["determine_correctness"]}.
                                """,
        "instruction_with_explanations": f"""You are to respond with one of: {{choices}}. 
                                {COMMON_INSTRUCTIONS["do_not_answer"]}. 
                                {COMMON_INSTRUCTIONS["determine_correctness"]}.
                                """,
    },
}

# =============================================================================
# UNIFIED INSTRUCTION GENERATION
# =============================================================================


def create_instruction(template_type: str, choices: str = None, with_explanations: bool = False) -> str:
    """Create instruction for any template type with or without explanations."""

    template = PROMPT_TEMPLATES[template_type]

    # Handle dynamic choices for categorical templates
    if template_type == "categorical" and choices:
        format_spec = f"Score: {SCORING_CONFIG['categorical']['score_format'].format(choices=choices)}"
        if with_explanations:
            instruction = f"""You are to respond with one of: {choices}. 
                {COMMON_INSTRUCTIONS["do_not_answer"]}. 
                {COMMON_INSTRUCTIONS["determine_correctness"].format(choices=choices)}.
                              """
        else:
            instruction = f"""You are to respond with ONLY one of: {choices}. 
                {COMMON_INSTRUCTIONS["only_choices"].format(choices=choices)}. 
                {COMMON_INSTRUCTIONS["do_not_answer"]}. 
                {COMMON_INSTRUCTIONS["determine_correctness"].format(choices=choices)}.
                              """
    else:
        format_spec = template["score_format"]
        instruction_key = "instruction_with_explanations" if with_explanations else "instruction_without_explanations"
        instruction = template[instruction_key]

    if with_explanations:
        return f"""{template["base"]}
                {instruction}

                You are to respond with the following format:

                {format_spec}
                Explanation: [Brief explanation of your reasoning]

                {template["examples_with"]}"""
    else:
        return f"""{template["base"]}
                {instruction}

                {template["examples_without"]}"""


# =============================================================================
# TEMPLATE MAPPINGS
# =============================================================================

TEMPLATE_TO_INSTRUCTION = {
    "true_false": create_instruction("categorical", SCORING_CONFIG["categorical"]["choices_2_class"], with_explanations=False),
    "true_false_uncertain": create_instruction("categorical", SCORING_CONFIG["categorical"]["choices_3_class"], with_explanations=False),
    "continuous": create_instruction("continuous", with_explanations=False),
    "likert": create_instruction("likert", with_explanations=False),
}

TEMPLATE_TO_INSTRUCTION_WITH_EXPLANATIONS = {
    "true_false": create_instruction("categorical", SCORING_CONFIG["categorical"]["choices_2_class"], with_explanations=True),
    "true_false_uncertain": create_instruction("categorical", SCORING_CONFIG["categorical"]["choices_3_class"], with_explanations=True),
    "continuous": create_instruction("continuous", with_explanations=True),
    "likert": create_instruction("likert", with_explanations=True),
}
