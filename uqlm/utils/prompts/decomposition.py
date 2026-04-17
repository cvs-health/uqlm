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
def get_claim_breakdown_prompt(response: str) -> str:
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

    claim_breakdown_prompt = f"""
    Please breakdown the following passage into independent fact pieces. 

    Step 1: For each sentence, you should break it into several fact pieces. Each fact piece should only contain one single independent fact. Normally the format of a fact piece is "subject + verb + object". If the sentence does not contain a verb, you can use "be" as the verb.

    Step 2: Do this for all the sentences. Output each piece of fact in one single line starting with ###. Do not include other formatting. 

    Step 3: Each atomic fact should be self-contained. Do not use pronouns as the subject of a piece of fact, such as he, she, it, this that, use the original subject whenever possible.

    Step 4: If the sentence does not contain any independent fact, you should output "### NONE".

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

    Example 4:
    Hi
    ### NONE

    Now it's your turn. Here is the passage: 

    {response}

    You should only return the final answer. Now your answer is:
    """

    return claim_breakdown_prompt


# The farquhar_2024_breakdown_template is a modified version of the prompt from "Detecting Hallucinations in Large Language Models Using Semantic Entropy"
# @article{farquhar2024detecting,
#       title={Detecting hallucinations in large language models using semantic entropy},
#       author={Farquhar, Sebastian and Kossen, Jannik and Kuhn, Lorenz and Gal, Yarin},
#       journal={Nature},
#       volume={630},
#       pages={625--630},
#       year={2024},
#       url={https://www.nature.com/articles/s41586-024-07421-0},
# }
def get_farquhar_2024_breakdown_prompt(response: str) -> str:
    """
    Claim decomposition prompt from Farquhar et al. (2024).

    Uses zero-shot instruction to list specific factual propositions. The original
    paper does not include few-shot examples in the decomposition prompt, resulting
    in coarser-grained claims that may retain pronouns from the original text.

    Parameters
    ----------
    response: str
        The response to be broken down into factual propositions.

    Returns
    -------
    str
        The prompt template for breaking down the response into factual propositions.
    """

    prompt = f"""Please list the specific factual propositions included in the paragraph below. Be complete and do not leave any factual claims out. Provide each claim as a separate sentence in a separate bullet point. Output each piece of fact in one single line starting with ###. Do not include other formatting.

{response}"""

    return prompt


# The mohri_2024_breakdown_template is a modified version of the prompt from "Language Models with Conformal Factuality Guarantees"
# @inproceedings{mohri2024language,
#       title={Language Models with Conformal Factuality Guarantees},
#       author={Mohri, Christopher and Hashimoto, Tatsunori},
#       booktitle={International Conference on Machine Learning},
#       year={2024},
#       url={https://arxiv.org/abs/2402.10978},
# }
def get_mohri_2024_breakdown_prompt(response: str) -> str:
    """
    Claim decomposition prompt adapted from Mohri & Hashimoto (2024).

    Instructs the LLM to break input into small, independent, non-overlapping claims.
    The original paper used JSONL output with confidence scores; this version is adapted
    to use the ### prefix format for compatibility with uqlm's claim parser.

    Parameters
    ----------
    response: str
        The response to be broken down into independent claims.

    Returns
    -------
    str
        The prompt template for breaking down the response into non-overlapping claims.
    """

    prompt = f"""Please breakdown the following input into a set of small, independent, non-overlapping claims. \
Make sure that each claim is small and non-overlapping. \
Output each claim on a single line starting with ###. Do not include any other formatting.

    Here are some examples:

    Example 1:
    Percy Bysshe Shelley was one of the major English Romantic poets, who is regarded by critics as among the finest lyric and philosophical poets in the English language. A radical in his poetry as well as his political and social views, Shelley did not see fame during his lifetime, but recognition of his achievements in poetry grew steadily following his death.
    ### Percy Bysshe Shelley was one of the major English Romantic poets.
    ### Percy Bysshe Shelley is regarded by critics as among the finest lyric poets in the English language.
    ### Percy Bysshe Shelley is regarded by critics as among the finest philosophical poets in the English language.
    ### Percy Bysshe Shelley was a radical in his poetry.
    ### Percy Bysshe Shelley was a radical in his political views.
    ### Percy Bysshe Shelley was a radical in his social views.
    ### Percy Bysshe Shelley did not see fame during his lifetime.
    ### Recognition of Percy Bysshe Shelley's achievements in poetry grew steadily following his death.

    Example 2:
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.
    ### The Eiffel Tower is a wrought-iron lattice tower.
    ### The Eiffel Tower is located on the Champ de Mars in Paris, France.
    ### The Eiffel Tower is named after the engineer Gustave Eiffel.
    ### Gustave Eiffel's company designed the Eiffel Tower.
    ### Gustave Eiffel's company built the Eiffel Tower from 1887 to 1889.

    Here is the input:

    {response}

    You should only return the final answer. Now your answer is:
    """

    return prompt


# The jiang_2024_breakdown_template is a modified version of the prompt from "Graph-based Uncertainty Metrics for Long-form Language Model Outputs"
# @misc{jiang2024graphbaseduncertaintymetricslongform,
#       title={Graph-based Uncertainty Metrics for Long-form Language Model Outputs},
#       author={Mingjian Jiang and Yangjun Ruan and Prasanna Sattigeri and Salim Roukos and Tatsunori Hashimoto},
#       year={2024},
#       eprint={2410.20783},
#       archivePrefix={arXiv},
#       primaryClass={cs.CL},
#       url={https://arxiv.org/abs/2410.20783},
# }
def get_jiang_2024_breakdown_prompt(response: str) -> str:
    """
    Claim decomposition prompt from Jiang et al. (2024).

    Instructs the LLM to deconstruct a paragraph into the smallest possible
    standalone, self-contained facts without semantic repetition. Designed for
    graph-based UQ pipelines where redundant claims would distort graph edges.

    Parameters
    ----------
    response: str
        The response to be broken down into non-redundant atomic facts.

    Returns
    -------
    str
        The prompt template for breaking down the response into minimal, non-redundant facts.
    """

    prompt = f"""Please deconstruct the following paragraph into the smallest possible standalone self-contained facts \
without semantic repetition. Output each fact on a single line starting with ###. Do not include any other formatting.

    Here are some examples:

    Example 1:
    Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science.
    ### Albert Einstein was born in Germany.
    ### Albert Einstein was a theoretical physicist.
    ### Albert Einstein developed the theory of relativity.
    ### The theory of relativity is one of the two pillars of modern physics.
    ### Albert Einstein's work influenced the philosophy of science.

    Example 2:
    The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups.
    ### The Great Wall of China is a series of fortifications.
    ### The Great Wall of China was built across the historical northern borders of China.
    ### The Great Wall of China was built by ancient Chinese states and Imperial China.
    ### The Great Wall of China was built as protection against nomadic groups.

    Here is the paragraph:

    {response}

    You should only return the final answer. Now your answer is:
    """

    return prompt


DECOMPOSITION_PROMPT_MAP = {"zhang_2025": get_claim_breakdown_prompt, "farquhar_2024": get_farquhar_2024_breakdown_prompt, "mohri_2024": get_mohri_2024_breakdown_prompt, "jiang_2024": get_jiang_2024_breakdown_prompt}


def get_factoid_breakdown_template(response: str) -> str:
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
