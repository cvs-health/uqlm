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


# The claim_brekadown_template is a modified version of the prompt from "Atomic Calibration of LLMs in Long-Form Generations"
# @misc{zhang2025atomiccalibrationllmslongform,
#       title={Atomic Calibration of LLMs in Long-Form Generations},
#       author={Caiqi Zhang and Ruihan Yang and Zhisong Zhang and Xinting Huang and Sen Yang and Dong Yu and Nigel Collier},
#       year={2025},
#       eprint={2410.13246},
#       archivePrefix={arXiv},
#       primaryClass={cs.CL},
#       url={https://arxiv.org/abs/2410.13246},
# }
def get_claim_breakdown_template(response: str) -> str:
    """
    Parameters
    ----------
    response: str
        The response to be broken down into fact pieces.

    Returns
    -------
    str
        The prompt template for breaking down the response into fact pieces.
    """

    claim_breakdown_template = f"""
    Please breakdown the following passage into independent fact pieces. 

    Step 1: For each sentence, you should break it into several fact pieces. Each fact piece should only contain one single independent fact. Normally the format of a fact piece is "subject + verb + object". If the sentence does not contain a verb, you can use "be" as the verb.

    Step 2: Do this for all the sentences. Output each piece of fact in one single line starting with ###. Do not include other formatting. 

    Step 3: Each atomic fact should be self-contained. Do not use pronouns as the subject of a piece of fact, such as he, she, it, this that, use the original subject whenever possible.

    Here are some examples:

    Example 1:
    Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969.
    ### Michael Collins was born on October 31, 1930.
    ### Michael Collins is retired.
    ### Michael Collins is an American.
    ### Michael Collins was an astronaut.
    ### Michael Collins was a test pilot.
    ### Michael Collins was the Command Module Pilot.
    ### Michael Collins was the Command Module Pilot for the Apollo 11 mission.
    ### Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969.

    Example 2:
    League of Legends (often abbreviated as LoL) is a multiplayer online battle arena (MOBA) video game developed and published by Riot Games. 
    ### League of Legends is a video game.
    ### League of Legends is often abbreviated as LoL.
    ### League of Legends is a multiplayer online battle arena.
    ### League of Legends is a MOBA video game.
    ### League of Legends is developed by Riot Games.
    ### League of Legends is published by Riot Games.

    Example 3:
    Emory University has a strong athletics program, competing in the National Collegiate Athletic Association (NCAA) Division I Atlantic Coast Conference (ACC). The university's mascot is the Eagle.
    ### Emory University has a strong athletics program.
    ### Emory University competes in the National Collegiate Athletic Association Division I.
    ### Emory University competes in the Atlantic Coast Conference.
    ### Emory University is part of the ACC.
    ### Emory University's mascot is the Eagle.

    Now it's your turn. Here is the passage: 

    {response}

    You should only return the final answer. Now your answer is:
    """

    return claim_breakdown_template


def get_entailment_template(claim: str, source_text: str) -> str:
    """
    Parameters
    ----------
    claim: str
        The claim to be evaluated.
    source_text: str
        The source text to be evaluated.

    Returns
    -------
    str
        The prompt template for evaluating the entailment of a claim and a source text.
    """

    entailment_template = """

    You are a helpful assistant that can evaluate the entailment of a claim and a source text.

    You will be given a claim and a source text. You need to evaluate if the claim is entailed by the source text.

    You should return one of the following categorizations:

    true - if the claim is entailed by the source text.
    false - if the claim is not entailed by the source text.

    Example:

    Source text:
    Emory University has a strong athletics program, competing in the National Collegiate Athletic Association (NCAA) Division I Atlantic Coast Conference (ACC). The university's mascot is the Eagle.

    Claim:
    Emory University is part of the ACC.

    Categorization:
    true

    Only return the categorization label (true or false). Do not include any other text.

    Source text:
    {source_text}

    Claim:
    {claim}

    Categorization:

    """
    return entailment_template


def get_factoid_template(response: str) -> str:
    """
    Parameters
    ----------
    response: str
        The response to be broken down into fact pieces.
    """

    factoid_template = f"""Please list the specific factual propositions included in the paragraph below. Be complete and do not leave any factual claims out. Provide each claim as a separate sentence in a separate bullet point. Do this for all the sentences. Output each piece of fact in one single line starting with ###. Do not include other formatting.

    Here are some examples:

    Example 1:
    Sir Paul McCartney is a renowned English musician, singer, and songwriter who gained worldwide fame as a member of The Beatles, one of the most influential and successful bands in the history of popular music. Born on June 18, 1942, in Liverpool, England, McCartney's career spans over six decades, during which he has become renowned not only for his role as a bass guitarist and vocalist with The Beatles but also for his songwriting partnership with John Lennon, which produced some of the most celebrated songs of the 20th century. After The Beatles disbanded in 1970, McCartney pursued a successful solo career and formed the band Wings with his late wife, Linda McCartney. His contributions to music have been recognized with numerous awards, including multiple Grammys and an induction into the Rock and Roll Hall of Fame both as a member of The Beatles and as a solo artist. Known for his melodic bass lines, versatile vocals, and enduring performances, McCartney continues to captivate audiences around the world with his artistry and dedication to his craft.
    ### Sir Paul McCartney is a renowned English musician and songwriter who gained worldwide fame as a member of The Beatles.
    ### He formed the band Wings after The Beatles disbanded and has had a successful solo career.

    Example 2:
    John Lennon was an iconic English musician, singer, and songwriter, best known as one of the founding members of The Beatles, the most commercially successful and critically acclaimed band in the history of popular music. Born on October 9, 1940, in Liverpool, Lennon grew up to become a pivotal figure in the cultural revolution of the 1960s. His songwriting partnership with Paul McCartney produced enduring classics that helped define the era. Known for his wit and outspoken personality, Lennon was also a passionate peace activist whose pursuit of social justice resonated with millions worldwide. After The Beatles disbanded, Lennon continued his musical career as a solo artist, producing hits like "Imagine," which became an anthem for peace. Tragically, his life was cut short on December 8, 1980, when he was assassinated in New York City, but his legacy continues to influence generations and inspire artists around the globe.
    ### John Lennon was an iconic English musician, singer, and songwriter. He was best known as one of the founding members of The Beatles.
    ### Born on October 9, 1940, in Liverpool, Lennon grew up to become a pivotal figure in the cultural revolution of the 1960s. His songwriting partnership with Paul McCartney produced enduring classics that helped define the era.
    ### Lennon was also a passionate peace activist whose pursuit of social justice resonated with millions worldwide. After The Beatles disbanded, Lennon continued his musical career as a solo artist, producing hits like "Imagine."

    Here is the paragraph:

    {response}

    You should only return the final answer. Now your answer is:
    """

    return factoid_template


def get_question_template(response: str, factoid_i: str, num_questions: int = 3) -> str:
    """
    Parameters
    ----------
    response: str
        The response to be broken down into fact pieces.
    factoid_i: str
        The factoid to be used as a question.
    num_questions: int
        The number of questions to generate.
    """

    question_template = f"""
    
    Following this text:
    
    {response}
    
    You see the sentence:

    {factoid_i}
    
    Generate a list of {num_questions} questions, that might have generated the sentence in the context of the preceding original text, as well as their answers. Please do not use specific facts that appear in the follow-up sentence when formulating the question. Make the questions and answers diverse. Avoid yes-no questions. The answers should not be a full sentence and as short as possible, e.g. only a name, place, or thing. Output each question in one single line starting with ###. Do not include other formatting.

    You should only return the final answer. Now your answer is:
    """

    return question_template

def get_answer_template(prompt: str, response: str, question: str) -> str:
    """
    Parameters
    ----------
    prompt: str
        The prompt to be used for generating the answer.
    response: str
        The response to be used for generating the answer.
    question: str
        The question to be used for generating the answer.
    """

    answer_template = f"""
    We are writing an answer to the question “{prompt}”. So far we have written:
    {response}
    The next sentence should be the answer to the following question:
    {question}
    Please answer this question. Do not answer in a full sentence. Answer with as few words as possible, e.g. only a name, place, or thing.

    You should only return the final answer. Now your answer is:
    """

    return answer_template