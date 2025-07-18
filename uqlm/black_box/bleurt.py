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


import importlib.resources as resources
import numpy as np
import os
import requests
import zipfile
from requests.exceptions import RequestException
from zipfile import BadZipFile

from typing import List, Optional

from uqlm.black_box.baseclass.similarity_scorer import SimilarityScorer
import time
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn


class BLEURTScorer(SimilarityScorer):
    def __init__(self) -> None:
        """
        Class for computing BLEURT Scores between original responses and candidates. For more on
        BLEURT, refer to Sellam et al.(2020) :footcite:`sellam2020bleurtlearningrobustmetrics`. Requires
        installation of `bleurt` package. Install using:
        `pip install pip install --user git+https://github.com/google-research/bleurt.git`

        Raises
        ------
        RuntimeError
            If there's an error downloading or initializing the BLEURT checkpoint
        """
        try:
            from bleurt.score import BleurtScorer
        except ImportError:
            raise ImportError(
                """
            The bleurt package is required to use BLEURTScorer but is not installed. Please install it using:\n
            `pip install git+https://github.com/google-research/bleurt.git`
            """
            )
        try:
            checkpoint = self._set_bleurt_checkpoint()
            self.bleurt_scorer = BleurtScorer(checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BLEURT scorer. Error: {str(e)}") from e

    def evaluate(self, responses: List[str], sampled_responses: List[List[str]], progress_bar: Optional[bool] = True) -> List[float]:
        """
        This method computes model-based text similarity metrics values for the provided pairs of texts.

        Parameters
        ----------
        responses : list of strings
            Original LLM response

        sampled_responses : list of list of strings
            Candidate responses to be compared to the original response

        progress_bar : bool, default=True
            If True, displays a progress bar while scoring responses

        Returns
        -------
        List of float
            Mean BLEURT scores
        """
        if progress_bar:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.completed}/{task.total}"), TimeElapsedColumn()) as progress:
                task = progress.add_task("- [black]Scoring responses with BLEURT...", total=len(responses))
                results = []
                for i in range(len(responses)):
                    score = self._compute_score(response=responses[i], candidates=sampled_responses[i])
                    results.append(score)
                    progress.update(task, advance=1)
                time.sleep(0.1)
                return results
        else:
            return [self._compute_score(response=responses[i], candidates=sampled_responses[i]) for i in range(len(responses))]

    def _compute_score(self, response: str, candidates: List[str]) -> float:
        """Compute BLEURT scores between a response and candidate responses"""
        duplicated_response = [response] * len(candidates)
        return np.mean(self.bleurt_scorer.score(references=duplicated_response, candidates=candidates))

    def _set_bleurt_checkpoint(self):
        """Sets up checkpoint"""
        resource_path = resources.files("uqlm.resources").joinpath("BLEURT-20")
        if not resource_path.is_dir():
            my_file_path = resources.files("uqlm.resources")
            zip_file_path = self._download(url="https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip", my_file_path=my_file_path)
            self._unzip(zip_file_path=zip_file_path, my_file_path=my_file_path)
        return resource_path

    @staticmethod
    def _download(url, my_file_path):
        """Download BLEURT checkpoint, unzip, and delete zip file"""
        zip_file_path = os.path.join(my_file_path, "BLEURT-20.zip")
        print(f"BLEURT checkpoint not found. Downloading to: {zip_file_path}")

        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(zip_file_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded zip file to: {zip_file_path}")
            else:
                print(f"Failed to download file. Status code: {response.status_code}")
                return
        except RequestException as e:
            print(f"Network error occurred while downloading BLEURT checkpoint: {str(e)}")
            return
        return zip_file_path

    @staticmethod
    def _unzip(zip_file_path, my_file_path):
        try:
            print(f"Unzipping: {zip_file_path}")
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(my_file_path)
                print(f"Unzipped files to: {my_file_path}")
        except BadZipFile as e:
            print(f"Error: The downloaded BLEURT zip file is corrupted: {str(e)}")
        except Exception as e:
            print(f"Unexpected error while extracting BLEURT zip file: {str(e)}")
        finally:
            if os.path.exists(zip_file_path):
                os.remove(zip_file_path)
                print(f"Deleted zip file: {zip_file_path}")
            return
