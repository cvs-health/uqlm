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

# import bert_score
from uqlm.similarity import BertScorer, BLEURTScorer, CosineScorer, MatchScorer


async def main():
    # Load data
    current_directory = os.getcwd()
    datafile_path = os.path.join("/".join(current_directory.split("/")[:-1]), "scorers/bsdetector_results_file.json")
    with open(datafile_path, "r") as f:
        data = json.load(f)

    responses = data["responses"]
    sampled_responses = data["sampled_responses"]

    store_results = dict()
    store_results.update({"responses": responses, "sampled_responses": sampled_responses})

    # 1. Bert Scorer
    bert = BertScorer()
    bert_result = bert.evaluate(responses=responses, sampled_responses=sampled_responses)

    store_results.update(
        {
            "bert_result": bert_result
            # 'F1': F1
        }
    )

    # 2. Bleurt Scorer
    bluert = BLEURTScorer()
    bluert_result = bluert.evaluate(responses=responses, sampled_responses=sampled_responses)

    store_results.update({"bluert_result": bluert_result})

    # 3. Cosine Similarity Scorer
    cosine = CosineScorer()
    cosine_result = cosine.evaluate(responses=responses, sampled_responses=sampled_responses)
    embeddings1, embeddings2 = [], []
    for i in range(len(responses)):
        embeddings1.append(cosine.model.encode([responses[i]] * len(sampled_responses[i])).tolist())
        embeddings2.append(cosine.model.encode(sampled_responses[i]).tolist())

    store_results.update({"cosine_result": cosine_result, "embeddings1": embeddings1, "embeddings2": embeddings2})

    # 4. Exact Match scorer
    match = MatchScorer()
    match_result = match.evaluate(responses=responses, sampled_responses=sampled_responses)

    store_results.update({"match_result": match_result})

    # Store results
    results_file = "similarity_results_file.json"
    with open(results_file, "w") as f:
        json.dump(store_results, f)


if __name__ == "__main__":
    main()
