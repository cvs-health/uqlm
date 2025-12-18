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
This module is used to store LLM prompt templates that can be used for various tasks.
"""


def get_question_template(claim: str):
    question_template = f"""
    Create a specific question that has this exact claim as its only correct answer. Return just the question with no extra text.

    Claim: {claim}
    """
    return question_template


def get_multiple_question_template(claim: str, num_questions: int = 2, response: str = None):
    if response:
        question_template = f"""
        Following this text: {response}
        You see the sentence: {claim}
        Generate a list of {num_questions} questions, that might have generated the sentence in the context of the preceding original text. Please do not use specific facts that appear in the follow-up sentence when formulating the questions. Avoid yes-no questions. The questions should have a concise, well-defined answer e.g. only a name, place, or thing. Output each question in one single line starting with ###. Do not include other formatting.

        Example:
        ### Here is the first question? ### Here is the second question?

        Now your response is:
        """
    else:
        question_template = f"""
    Create a list of {num_questions} distinct questions that are answered by the below factoid. Each question should have a unique answer that is contained in the factoid. Output each question in one single line starting with ###. Do not include other formatting.

    Factoid: {claim}

    Example:
    ### Here is the first question? ### Here is the second question?

    Now your response is:
    """
    return question_template


def get_answer_template(claim_question: str, original_question: str = None, original_response: str = None) -> str:
    """
    Parameters
    ----------
    original_question: str
        The original question to be used for generating the answer.
    original_response: str
        The original response to be used for generating the answer.
    claim_question: str
        The claim question to be used for generating the answer.
    """
    prefix = ""
    if original_question:
        prefix += f"""We are working on an answer to this question: "{original_question}."\n\n"""
    if original_response:
        answer_template = f"""
    So far we have written:
    {original_response}
    The next sentence should be the answer to the following question: {claim_question}
    Please answer only this question. Do not answer in a full sentence. Answer with as few words as possible, e.g. only a name, place, or thing.
    """
    else:
        answer_template = f"""**Your task**: With this in mind, consider only the following question and answer with as few words as possible:\n\n{claim_question}\n\nNow your answer is:"""
    return prefix + answer_template


# def get_question_template(response: str, factoid_i: str, num_questions: int = 3) -> str:
#     """
#     Parameters
#     ----------
#     response: str
#         The response to be broken down into fact pieces.
#     factoid_i: str
#         The factoid to be used as a question.
#     num_questions: int
#         The number of questions to generate.
#     """

#     question_template = f"""

#     Following this text:

#     {response}

#     You see the sentence:

#     {factoid_i}

#     Generate a list of {num_questions} questions, that might have generated the sentence in the context of the preceding original text. Please do not use specific facts that appear in the follow-up sentence when formulating the question. Make the questions and answers diverse. Avoid yes-no questions. Output each question in one single line starting with ###. Do not include other formatting.

#     You should only return the final answer. Now your answer is:
#     """

# return question_template

# def get_question_template(
#     # response: str,
#     factoid_i: str,
#     # num_questions: int = 3
# ) -> str:
#     """
#     Parameters
#     ----------
#     response: str
#         The response to be broken down into fact pieces.
#     factoid_i: str
#         The factoid to be used as a question.
#     num_questions: int
#         The number of questions to generate.
#     """

#     question_template = f"""
#     You are an expert at creating precise, targeted questions. Your task is to generate a question for which a given atomic claim is the unique correct answer.

#     I will provide you with an atomic claim - a single, specific statement of fact. Your response must contain ONLY the question you've created, with no explanations or additional text.

#     Create a question that meets these criteria:

#     - The given claim is the complete and correct answer to your question
#     - The question should have ONLY this claim as its answer (not a broader or narrower answer)
#     - The question should NOT have multiple acceptable answers
#     - The question should be clear and unambiguous
#     - The question should be specific enough that someone knowledgeable in the domain would converge on this exact claim
#     - The question should not give away the answer in its phrasing
#     - Return ONLY the question with no additional text

#     ## Examples
#     Atomic Claim: "The Eiffel Tower was completed in 1889."

#     - Poor Question: "Was the Eiffel Tower completed in 1889 or 1890?" (Gives away the answer)
#     - Poor Question: "What structure was completed in 1889" (Eiffel Tower is not the only valid answer)
#     - Good Question: "In what year was construction of the Eiffel Tower fully completed?" (Specific, doesn't hint at the answer, and uniquely elicits the claim)

#     For the following atomic claim, generate a question, and return only that question, that uniquely elicits this claim as its answer:

#     {factoid_i}
#     """
#     return question_template


# def get_question_template(
#     # response: str,
#     factoid_i: str,
#     # num_questions: int = 3
# ) -> str:
#     """
#     Parameters
#     ----------
#     response: str
#         The response to be broken down into fact pieces.
#     factoid_i: str
#         The factoid to be used as a question.
#     num_questions: int
#         The number of questions to generate.
#     """

#     question_template = f"""
#     You are an expert at creating precise questions. Generate a question for which the following claim is the unique correct answer. Return ONLY the question with no additional text.

#     Guidelines:
#     - The claim must be the complete and correct answer
#     - The question should have only this claim as its answer
#     - Avoid giving away the answer in your phrasing
#     - Make the question clear and specific

#     Example:
#     Claim: "The Eiffel Tower was completed in 1889."
#     Good Question: "In what year was construction of the Eiffel Tower fully completed?"

#     Claim: {factoid_i}
#     """
#     return question_template
