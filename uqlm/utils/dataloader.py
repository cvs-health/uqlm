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

import pandas as pd
from typing import List, Optional, Union
from datasets import load_dataset, concatenate_datasets
from datasets import disable_progress_bars
import importlib.util
import re
import ast
import numpy as np
import warnings
from copy import deepcopy

"""This module uses the _dataset_default_params dict to control what datasets load_example_dataset can load and how they are loaded.

You can add new datasets to the loader by adding a new entry in the _dataset_default_params dict like:

'my_dataset_name': {

    'load_params': {'path': 'hf_hub_org_name/org_dataset_name', # HF dataset ref
                    'name': 'if_hf_dataset_has_multi_files', # needed for specific datasets
                    'split': 'train'}, # optional
    'extra_processing': {} # optional, see examples in _dataset_default_params dict

}
"""
_dataset_default_params = {
    "ai2_arc": {
        "load_params": {
            "path": "allenai/ai2_arc",  # HF Hub dataset name
            "name": "ARC-Easy",  # HF Hub filename
            "split": "test",
        },  # HF Hub dataset split
        "extra_processing": {
            "rename_columns": {"answerKey": "answer"},  # renaming is always the first operation
            "strip_whitespace": ["answer"],  # notice we're referencing the renamed col name
            "to_upper": ["answer"],
            "combine_question_and_choices": {"question_col": "question", "choice_col": "choices", "choice_text_col": "text", "choice_label_col": "label"},
            "subset_columns": ["question", "answer"],
        },
    },
    "csqa": {"load_params": {"path": "skrishna/CSQA_preprocessed", "split": "train"}, "extra_processing": {"rename_columns": {"question": "q_only", "answerKey": "answer", "inputs": "question"}, "to_upper": ["answer"], "subset_columns": ["question", "answer"]}},
    "gsm8k": {
        "load_params": {"path": "openai/gsm8k", "name": "main", "split": "train"},
        "extra_processing": {
            "regex_filters": [
                {
                    "pattern": r"#### ([-+]?\d*\.\d+|[-+]?\d+)",  # regex pattern
                    "col": "answer",  # dataset col to apply pattern to
                    "operation": "search",  # type of regex operation
                    "group": 1,
                }
            ],  # capture group desired
            "subset_columns": ["question", "answer"],
        },
    },
    "nq_open": {"load_params": {"path": "google-research-datasets/nq_open", "split": "validation"}, "extra_processing": {"to_lower": ["answer"], "subset_columns": ["question", "answer"]}},
    "popqa": {"load_params": {"path": "akariasai/PopQA", "split": "test"}, "extra_processing": {"rename_columns": {"possible_answers": "answer"}, "to_lower": ["answer"], "subset_columns": ["question", "answer"]}},
    "svamp": {
        "load_params": {"path": "Chilled/SVAMP"},
        "extra_processing": {
            "concat": "all",
            "rename_columns": {"question_concat": "question", "Answer": "answer"},
            "regex_filters": [
                {
                    "pattern": r"#### ([-+]?\d*\.\d+|[-+]?\d+)",  # get numbers only (like gsm8k)
                    "col": "answer",
                    "operation": "search",
                    "group": 1,
                }
            ],
            "subset_columns": ["question", "answer"],
        },
    },
    "factscore": {"load_params": {"path": "dskar/FActScore", "split": "test"}, "extra_processing": {}},
    "hotpotqa": {"load_params": {"path": "hotpotqa/hotpot_qa", "name": "distractor", "split": "validation"}, "extra_processing": {"subset_columns": ["question", "answer"]}},
    "simpleqa": {"load_params": {"path": "google/simpleqa-verified", "split": "eval"}, "extra_processing": {"subset_columns": ["question", "answer"], "rename_columns": {"problem": "question"}}},
    "livecodebench": {"load_params": {"path": "livecodebench/code_generation_lite", "split": "test"}, "extra_processing": {"subset_columns": ["question_title", "question_content", "platform", "question_id", "starter_code", "public_test_cases", "metadata", "difficulty"]}},
    "factscore-stem-geo": {"load_params": {"loader": "_load_factscore_stem_geo_dataset"}, "extra_processing": {}},
}

USER_AGENT = "uqlm/0.6.0 (https://github.com/cvs-health/uqlm)"


def list_dataset_names() -> list:
    """
    List all available example dataset names in uqlm.

    Returns
    -------
    list
        A list of available datasets.

    Example
    -------
    >>> from uqlm.utils.dataloader import list_dataset_names
    >>> list_dataset_names()
    ['ai2_arc', 'csqa', 'gsm8k', 'nq_open', 'popqa', 'svamp', 'factscore', 'hotpotqa', 'simpleqa', 'livecodebench', 'factscore-stem-geo']
    """
    return list(_dataset_default_params.keys())


def load_example_dataset(name: str, n: int = None, cols: Optional[Union[list, str]] = None, split: Optional[str] = None) -> pd.DataFrame:
    """
    Load a dataset for testing purposes.

    Parameters
    ----------
    name : str
        The name of the dataset to load. Must be one of "svamp", "gsm8k", "ai2_arc",
        "csqa", "nq_open", "popqa", "factscore", "hotpotqa", "simpleqa",
        "livecodebench", "factscore-stem-geo"

    n : int, optional
        Number of rows to load from the dataset. Ignored for "factscore-stem-geo",
        which always returns the longest 100 articles for each of four categories."
    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Example
    -------
    >>> from uqlm.utils.dataloader import load_example_dataset
    >>> df = load_example_dataset("gsm8k", n=1000)
    >>> df.shape
    (1000, 2)
    """
    dataset_dict = deepcopy(_dataset_default_params)
    if name in dataset_dict.keys():  # loads from huggingface hub
        disable_progress_bars()  # disable hf tqdm bars b/c it's a little ugly
        print(f"Loading dataset - {name}...")
        if dataset_dict[name]["load_params"].get("loader") == "_load_factscore_stem_geo_dataset":
            if isinstance(n, int):
                print("""Note: the 'n' parameter is not used for 'factscore-stem-geo' — the longest 100 articles will be returned for four categories: chemical elements, nerves in the human body, mountains on Earth, and scientific laws.""")
            print("Fetching Wikipedia articles — this may take a few minutes...")
            df = _load_factscore_stem_geo_dataset()
            if cols:
                df = _dataset_processing(df=df, subset_columns=cols)
            print("Dataset ready!")
            return df
        if split:
            dataset_dict[name]["load_params"]["split"] = split
        ds = load_dataset(**dataset_dict[name]["load_params"])
        print("Processing dataset...")
        extras = dataset_dict[name].get("extra_processing", dict())
        if extras:
            if extras.get("concat"):  # combine different splits into one
                split = extras.get("concat")
                if split == "all":  # combine splits into one dataset
                    ds = concatenate_datasets([ds[s] for s in ds])
                extras.pop("concat")
        df = ds.to_pandas()
        if cols:
            extras["subset_columns"] = cols
        if extras:
            df = _dataset_processing(df=df, **extras)  # data wrangling on single df
        if isinstance(n, int):
            df = df.iloc[:n]
        if name == "popqa":
            df["answer"] = [ast.literal_eval(a) for a in df["answer"]]
        print("Dataset ready!")
        return df
    else:
        raise FileNotFoundError(f"uqlm could not find the dataset '{name}'.\nPlease use `uqlm.utils.dataloader.list_dataset_names()` for available sample datasets.")


def _dataset_processing(df: pd.DataFrame, rename_columns: dict = None, subset_columns: list = None, to_upper: list = None, to_lower: list = None, combine_question_and_choices: dict = None, strip_non_numeric: list = None, strip_whitespace: list = None, regex_filters: list[dict] = None) -> pd.DataFrame:
    """
    Process a dataset with various operations.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to process.
    rename_columns : dict, optional
        A dictionary mapping old column names to new column names.
    subset_columns : list, optional
        A list of columns to keep in the dataframe.
    to_upper : list, optional
        A list of columns whose string values should be converted to uppercase.
    to_lower: list, optional
        A list of columns whose string values should be converted to lowercase.
    combine_question_and_choices : dict, optional
        A dictionary with parameters to combine question and question choice columns.
    strip_non_numeric : list, optional
        A list of columns from which to strip non-numeric characters.
    strip_whitespace : list, optional
        A list of columns from which to strip whitespace characters.
    regex_filters: list[dict]
        A list of dictionaries like `{'pattern':r'', 'col':''}` to describe regex transformations to apply on the dataset.
    Returns
    -------
    pd.DataFrame
        The processed dataframe.

    Raises
    ------
    TypeError
        If the input `df` is not a pandas DataFrame.

    Example
    -------
    >>> df = pd.DataFrame({"A": ["a", "b", "c"], "B": ["1", "2", "3"]})
    >>> _dataset_processing(df, rename_columns={"A": "a"}, to_upper=["a"])
       a  B
    0  A  1
    1  B  2
    2  C  3
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Dataset processing requires 'pd.DataFrame' but received '{type(df)}'")

    if rename_columns:
        df = df.rename(columns=rename_columns)
    if strip_non_numeric:
        for col in strip_non_numeric:
            df[col] = df[col].apply(lambda x: "".join(c for c in x if c.isdigit()))
    if strip_whitespace:
        for col in strip_whitespace:
            df[col] = df[col].str.replace(" ", "")
    if to_upper:
        for col in to_upper:
            if isinstance(df[col][0], (list, np.ndarray)):
                df[col] = df[col].apply(lambda x: [s.upper() for s in x])
            else:
                df[col] = df[col].str.upper()
    if to_lower:
        for col in to_lower:
            if isinstance(df[col][0], (list, np.ndarray)):
                df[col] = df[col].apply(lambda x: [s.lower() for s in x])
            else:
                df[col] = df[col].str.lower()
    if combine_question_and_choices:
        df = _combine_question_and_choices(df, **combine_question_and_choices)
    if regex_filters:
        for rfilter in regex_filters:
            if rfilter["operation"] == "search":
                if not rfilter.get("group", None):
                    rfilter["group"] = 0

                df[rfilter["col"]] = df[rfilter["col"]].apply(lambda x: re.search(rfilter["pattern"], x).group(rfilter["group"]) if re.search(rfilter["pattern"], x) else x)
    if subset_columns:
        cols = subset_columns
        if isinstance(cols, (list, str)):
            if isinstance(cols, str):
                cols = [cols]
            cols_in_df = [x for x in cols if x in df.columns]
            df = df[cols_in_df]
            if df.shape[1] != len(cols):
                print("WARNING: some specified columns not found in dataset...")

    return df


def _combine_question_and_choices(df: pd.DataFrame, question_col: str, choice_col: Union[str, list] = None, choice_text_col: str = None, choice_label_col: str = None, save_original_question: bool = False) -> pd.DataFrame:
    """
    Combine question and choices columns in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to process.
    question_col : str
        The name of the question column.
    choice_col : Union[str, list], optional
        The name(s) of the choice column(s). If a string, it should be the name of a column containing dictionaries.
        If a list, it should be the names of columns containing choices.
    choice_text_col : str, optional
        The name of the choice text column, used when `choice_col` is a string.
    choice_label_col : str, optional
        The name of the choice label column, used when `choice_col` is a string.
    save_original_question : bool, optional
        Whether to save the original question column as 'original_question'.

    Returns
    -------
    pd.DataFrame
        The processed dataframe with combined question and choices.

    Raises
    ------
    TypeError
        If `choice_col` is not a string or a list.

    Example
    -------
    >>> df = pd.DataFrame({"question": ["What is the capital of France?", "What is 2+2?"], "choices": [{"text": ["Paris", "London"], "label": ["A", "B"]}, {"text": ["3", "4"], "label": ["A", "B"]}]})
    >>> _combine_question_and_choices(df, question_col="question", choice_col="choices", choice_text_col="text", choice_label_col="label")
    >>> df
                               question
    0  What is the capital of France? A) Paris B) London
    1                          What is 2+2? A) 3 B) 4
    """
    if save_original_question and question_col == "question":
        df["original_question"] = df["question"].copy()
    if isinstance(choice_col, str):
        if isinstance(df[choice_col][0], dict):  # example of this format is allenai/ai2_arc
            df["question"] = df[question_col] + " " + df[choice_col].apply(lambda x: " ".join([f"{label}) {text}" for text, label in zip(x[choice_text_col], x[choice_label_col])]))
    elif isinstance(choice_col, list):
        # TODO when a dataset needs this... ex would be where cols are like answerA, answerB
        pass
    else:
        raise TypeError(f"'choice_col' must be str or list, but received '{type(choice_col)}'")
    return df


FACTSCORE_STEM_GEO_ENTITIES = {
    "nerve": [
        "Abdominal aortic plexus",
        "Abducens nerve",
        "Accessory nerve",
        "Accessory obturator nerve",
        "Alderman's nerve",
        "Anococcygeal nerve",
        "Ansa cervicalis",
        "Anterior interosseous nerve",
        "Anterior superior alveolar nerve",
        "Auerbach's plexus",
        "Auriculotemporal nerve",
        "Axillary nerve",
        "Brachial plexus",
        "Buccal branch of the facial nerve",
        "Buccal nerve",
        "Cardiac plexus",
        "Cavernous nerves",
        "Cavernous plexus",
        "Celiac ganglia",
        "Cervical branch of the facial nerve",
        "Cervical plexus",
        "Chorda tympani",
        "Ciliary ganglion",
        "Coccygeal nerve",
        "Cochlear nerve",
        "Common fibular nerve",
        "Common palmar digital nerves of median nerve",
        "Deep branch of the radial nerve",
        "Deep fibular nerve",
        "Deep petrosal nerve",
        "Deep temporal nerves",
        "Diagonal band of Broca",
        "Digastric branch of facial nerve",
        "Dorsal branch of ulnar nerve",
        "Dorsal nerve of clitoris",
        "Dorsal nerve of the penis",
        "Dorsal scapular nerve",
        "Esophageal plexus",
        "Ethmoidal nerves",
        "External laryngeal nerve",
        "External nasal nerve",
        "Facial nerve",
        "Femoral nerve",
        "Frontal nerve",
        "Gastric plexuses",
        "Geniculate ganglion",
        "Genital branch of genitofemoral nerve",
        "Genitofemoral nerve",
        "Glossopharyngeal nerve",
        "Greater auricular nerve",
        "Greater occipital nerve",
        "Greater petrosal nerve",
        "Hepatic plexus",
        "Hypoglossal nerve",
        "Iliohypogastric nerve",
        "Ilioinguinal nerve",
        "Inferior alveolar nerve",
        "Inferior anal nerves",
        "Inferior cardiac nerve",
        "Inferior cervical ganglion",
        "Inferior gluteal nerve",
        "Inferior hypogastric plexus",
        "Inferior mesenteric plexus",
        "Inferior palpebral nerve",
        "Infraorbital nerve",
        "Infraorbital plexus",
        "Infratrochlear nerve",
        "Intercostal nerves",
        "Intercostobrachial nerve",
        "Intermediate cutaneous nerve",
        "Internal carotid plexus",
        "Internal laryngeal nerve",
        "Interneuron",
        "Jugular ganglion",
        "Lacrimal nerve",
        "Lateral cord",
        "Lateral cutaneous nerve of forearm",
        "Lateral cutaneous nerve of thigh",
        "Lateral pectoral nerve",
        "Lateral plantar nerve",
        "Lateral pterygoid nerve",
        "Lesser occipital nerve",
        "Lingual nerve",
        "Long ciliary nerves",
        "Long root of the ciliary ganglion",
        "Long thoracic nerve",
        "Lower subscapular nerve",
        "Lumbar nerves",
        "Lumbar plexus",
        "Lumbar splanchnic nerves",
        "Lumboinguinal nerve",
        "Lumbosacral plexus",
        "Lumbosacral trunk",
        "Mandibular nerve",
        "Marginal mandibular branch of facial nerve",
        "Masseteric nerve",
        "Maxillary nerve",
        "Medial cord",
        "Medial cutaneous nerve of arm",
        "Medial cutaneous nerve of forearm",
        "Medial cutaneous nerve",
        "Medial pectoral nerve",
        "Medial plantar nerve",
        "Medial pterygoid nerve",
        "Median nerve",
        "Meissner's plexus",
        "Mental nerve",
        "Middle cardiac nerve",
        "Middle cervical ganglion",
        "Middle meningeal nerve",
        "Motor nerve",
        "Muscular branches of the radial nerve",
        "Musculocutaneous nerve",
        "Mylohyoid nerve",
        "Nasociliary nerve",
        "Nasopalatine nerve",
        "Nerve of pterygoid canal",
        "Nerve to obturator internus",
        "Nerve to quadratus femoris",
        "Nerve to the Piriformis",
        "Nerve to the stapedius",
        "Nerve to the subclavius",
        "Nervus intermedius",
        "Nervus spinosus",
        "Nodose ganglion",
        "Obturator nerve",
        "Oculomotor nerve",
        "Olfactory nerve",
        "Ophthalmic nerve",
        "Optic nerve",
        "Otic ganglion",
        "Ovarian plexus",
        "Palatine nerves",
        "Palmar branch of the median nerve",
        "Palmar branch of ulnar nerve",
        "Pancreatic plexus",
        "Patellar plexus",
        "Pelvic splanchnic nerves",
        "Perforating cutaneous nerve",
        "Perineal branches of posterior femoral cutaneous nerve",
        "Perineal nerve",
        "Petrous ganglion",
        "Pharyngeal branch of vagus nerve",
        "Pharyngeal branches of glossopharyngeal nerve",
        "Pharyngeal nerve",
        "Pharyngeal plexus",
        "Phrenic nerve",
        "Phrenic plexus",
        "Posterior auricular nerve",
        "Posterior branch of spinal nerve",
        "Posterior cord",
        "Posterior cutaneous nerve of arm",
        "Posterior cutaneous nerve of forearm",
        "Posterior cutaneous nerve of thigh",
        "Posterior scrotal nerves",
        "Posterior superior alveolar nerve",
        "Proper palmar digital nerves of median nerve",
        "Prostatic plexus (nervous)",
        "Pterygopalatine ganglion",
        "Pudendal nerve",
        "Pudendal plexus",
        "Pulmonary branches of vagus nerve",
        "Radial nerve",
        "Recurrent laryngeal nerve",
        "Renal plexus",
        "Sacral plexus",
        "Sacral splanchnic nerves",
        "Saphenous nerve",
        "Sciatic nerve",
        "Semilunar ganglion",
        "Sensory nerve",
        "Short ciliary nerves",
        "Sphenopalatine nerves",
        "Splenic plexus",
        "Stylohyoid branch of facial nerve",
        "Subcostal nerve",
        "Submandibular ganglion",
        "Suboccipital nerve",
        "Superficial branch of the radial nerve",
        "Superficial fibular nerve",
        "Superior cardiac nerve",
        "Superior cervical ganglion",
        "Superior ganglion of glossopharyngeal nerve",
        "Superior ganglion of vagus nerve",
        "Superior gluteal nerve",
        "Superior hypogastric plexus",
        "Superior labial nerve",
        "Superior laryngeal nerve",
        "Superior lateral cutaneous nerve of arm",
        "Superior mesenteric plexus",
        "Superior rectal plexus",
        "Supraclavicular nerves",
        "Supraorbital nerve",
        "Suprarenal plexus",
        "Suprascapular nerve",
        "Supratrochlear nerve",
        "Sural nerve",
        "Sympathetic trunk",
        "Temporal branches of the facial nerve",
        "Third occipital nerve",
        "Thoracic aortic plexus",
        "Thoracic splanchnic nerves",
        "Thoraco-abdominal nerves",
        "Thoracodorsal nerve",
        "Tibial nerve",
        "Transverse cervical nerve",
        "Trigeminal nerve",
        "Trochlear nerve",
        "Tympanic nerve",
        "Ulnar nerve",
        "Upper subscapular nerve",
        "Uterovaginal plexus",
        "Vagus nerve",
        "Ventral ramus",
        "Vesical nervous plexus",
        "Vestibular nerve",
        "Vestibulocochlear nerve",
        "Zygomatic branches of facial nerve",
        "Zygomatic nerve",
        "Zygomaticofacial nerve",
        "Zygomaticotemporal nerve",
    ],
    "scientific law or theorem": [
        "Abel's theorem",
        "Ariadne's thread",
        "Amdahl's law",
        "Ampère's circuital law",
        "Archie's law",
        "Archimedes's principle",
        "Axiom of Archimedes",
        "Arrhenius equation",
        "Avogadro's law",
        "Basquin's Law of Fatigue",
        "Bell's theorem",
        "Benford's law",
        "Beer–Lambert law",
        "Bernoulli's principle",
        "Bernoulli's equation",
        "Biot–Savart law",
        "Birch's law",
        "Bogoliubov–Born–Green–Kirkwood–Yvon hierarchy",
        "Bogoliubov transformation",
        "Boltzmann equation",
        "Born's law",
        "Boyle's law",
        "Bragg's Law",
        "Bradford's law",
        "Bruun Rule",
        "Buys Ballot's law",
        "Byerlee's law",
        "Carnot's theorem",
        "Cauchy's integral formula",
        "Cauchy–Riemann equations",
        "Cayley–Hamilton theorem",
        "Charles's law",
        "Chandrasekhar limit",
        "Church–Turing thesis",
        "Coulomb's law",
        "Law of Charles and Gay-Lussac",
        "Clifford's theorem",
        "Clifford's circle theorems",
        "Curie's law",
        "Curie–Weiss law",
        "D'Alembert's paradox",
        "D'Alembert's principle",
        "Dalton's law of partial pressure",
        "Darcy's law",
        "De Bruijn–Erdős theorem",
        "De Morgan's law",
        "Dermott's law",
        "Descartes's theorem",
        "Dirac equation",
        "Dirac delta function",
        "Dirac comb",
        "Dirac spinor",
        "Dirac operator",
        "Drake equation",
        "Doppler effect",
        "Ehrenfest's theorem",
        "Einstein's general theory of relativity",
        "Einstein's special theory of relativity",
        "El-Sayed rule",
        "Erdős–Anning theorem",
        "Erdős–Beck theorem",
        "Erdős–Gallai theorem",
        "Erdős–Kac theorem",
        "Erdős–Ko–Rado theorem",
        "Erdős–Nagy theorem",
        "Erdős–Rado theorem",
        "Erdős–Stone theorem",
        "Erdős–Szekeres theorem",
        "Erdős–Szemerédi theorem",
        "Euclid's theorem",
        "Euler's theorem",
        "Faraday's law of induction",
        "Faraday's law of electrolysis",
        "Faxén's law",
        "Fermat's principle",
        "Fermat's Last Theorem",
        "Fermat's little theorem",
        "Fermi paradox",
        "Fermi's golden rule",
        "Fermi acceleration",
        "Fermi hole",
        "Fermionic field",
        "Fermi level",
        "Fick's law of diffusion",
        "Fitts's law",
        "Fourier's law",
        "Gauss's law",
        "Gauss's law for magnetism",
        "Gauss's principle of least constraint",
        "Gauss's digamma theorem",
        "Gauss's hypergeometric theorem",
        "Gaussian function",
        "Gay-Lussac's law",
        "Gibbs–Helmholtz equation",
        "Gödel's incompleteness theorems",
        "Graham's law",
        "Green's law",
        "Grimm's law",
        "Gustafson's law",
        "Heisenberg's uncertainty principle",
        "Haüy's law of rational indices",
        "Haüy's law of symmetry",
        "Heaps' law",
        "Hellmann–Feynman theorem",
        "Henry's law",
        "Hertz observations",
        "Hess's law",
        "Hilbert's basis theorem",
        "Hilbert's axioms",
        "Hilbert function",
        "Hilbert's irreducibility theorem",
        "Hilbert's syzygy theorem",
        "Hilbert's Theorem 90",
        "Hilbert's theorem",
        "Hohenberg–Kohn theorem",
        "Helmholtz's theorems",
        "Helmholtz theorem",
        "Helmholtz free energy",
        "Helmholtz decomposition",
        "Helmholtz equation",
        "Helmholtz resonance",
        "Hollomon's law",
        "Hooke's law",
        "Hopkinson's law",
        "Hubble's law",
        "Hund's rules",
        "Huygens–Fresnel principle",
        "Joule's laws",
        "Jurin's law",
        "Kasha's rule",
        "Kepler's laws of planetary motion",
        "Kirchhoff's laws",
        "Kopp's law",
        "Larmor formula",
        "Leidenfrost effect",
        "Lagrangian point",
        "Lagrange reversion theorem",
        "Lagrange polynomial",
        "Lagrange's four-square theorem",
        "Lagrange's theorem",
        "Lagrange's theorem (group theory)",
        "Lagrange invariant",
        "Lagrange multiplier",
        "Lambert's cosine law",
        "Lamm equation",
        "Langmuir equation",
        "Laplace transform",
        "Laplace's equation",
        "Laplace operator",
        "Laplace distribution",
        "Laplace invariant",
        "Laplace expansion",
        "Laplace principle",
        "Laplace limit",
        "Le Chatelier's principle",
        "Leibniz's law",
        "Lenz's law",
        "Leonard–Merritt mass estimator",
        "l'Hôpital's rule",
        "Llinás's law",
        "Ludwik's law",
        "Mach principle",
        "Mach reflection",
        "Marconi's law",
        "Markovnikov's rule",
        "Maupertuis's principle",
        "Maxwell's equations",
        "Maxwell relations",
        "McCulloch's Iron Laws of Conferences",
        "Mendelian inheritance",
        "Mendel's laws",
        "Metcalfe's law",
        "Mikheyev–Smirnov–Wolfenstein effect",
        "Milner–Rado paradox",
        "Minkowski's theorem",
        "Mitscherlich's law",
        "Moore's law",
        "Nash embedding theorem",
        "Nash equilibrium",
        "Nernst equation",
        "Newton's law of cooling",
        "Newton's law of universal gravitation",
        "Newton's laws of motion",
        "Niven's theorem",
        "Noether's theorem",
        "Nyquist–Shannon sampling theorem",
        "Occam's razor",
        "Ohm's law",
        "Osipkov–Merritt model",
        "Ostwald dilution law",
        "Paley–Wiener theorem",
        "Pareto distribution",
        "Pareto efficiency",
        "Pareto index",
        "Pareto principle",
        "Pascal's law",
        "Pascal's theorem",
        "Pauli exclusion principle",
        "Peano axioms",
        "Planck's law",
        "Poincaré–Bendixson theorem",
        "Poincaré–Birkhoff–Witt theorem",
        "Poincaré–Hopf theorem",
        "Poincaré recurrence theorem",
        "Poincaré conjecture",
        "Poincaré lemma",
        "Poiseuille's law",
        "Poisson distribution",
        "Poisson's equation",
        "Price's theorem",
        "Ptolemy's theorem",
        "Pythagorean theorem",
        "Raman scattering",
        "Rado's theorem",
        "Ramanujan–Nagell equation",
        "Raoult's law",
        "Riemann zeta function",
        "Riemann hypothesis",
        "Riemann integral",
        "Riemann lemma",
        "Riemannian manifold",
        "Riemann sphere",
        "Riemann theta function",
        "Rolle's theorem",
        "Saha ionization equation",
        "Schrödinger equation",
        "Seebeck effect",
        "Sérsic's law",
        "Snell's law",
        "Sokolov–Ternov effect",
        "Sommerfeld–Kossel displacement law",
        "Stefan–Boltzmann law",
        "Steno's law",
        "Stokes' law",
        "Stoletov's law",
        "Swift's law",
        "Tarski's undefinability theorem",
        "Tarski's axioms",
        "Thales's theorem",
        "Titius–Bode law",
        "Torricelli's law",
        "Umov effect",
        "Van der Waals equation",
        "Vlasov equation",
        "Voce's law",
        "Von Neumann bicommutant theorem",
        "Von Neumann entropy",
        "von Neumann paradox",
        "Von Neumann ergodic theorem",
        "Von Neumann universe",
        "Von Neumann neighborhood",
        "Von Neumann's trace inequality",
        "Weinberg–Witten theorem",
        "Weyl character formula",
        "Wien's law",
        "Wiener–Khinchin theorem",
        "Young–Laplace equation",
        "Zener-Hollomon law",
        "Zipf's law",
    ],
    "chemical element": [
        "Hydrogen",
        "Helium",
        "Lithium",
        "Beryllium",
        "Boron",
        "Carbon",
        "Nitrogen",
        "Oxygen",
        "Fluorine",
        "Neon",
        "Sodium",
        "Magnesium",
        "Aluminium",
        "Silicon",
        "Phosphorus",
        "Sulfur",
        "Chlorine",
        "Argon",
        "Potassium",
        "Calcium",
        "Scandium",
        "Titanium",
        "Vanadium",
        "Chromium",
        "Manganese",
        "Iron",
        "Cobalt",
        "Nickel",
        "Copper",
        "Zinc",
        "Gallium",
        "Germanium",
        "Arsenic",
        "Selenium",
        "Bromine",
        "Krypton",
        "Rubidium",
        "Strontium",
        "Yttrium",
        "Zirconium",
        "Niobium",
        "Molybdenum",
        "Technetium",
        "Ruthenium",
        "Rhodium",
        "Palladium",
        "Silver",
        "Cadmium",
        "Indium",
        "Tin",
        "Antimony",
        "Tellurium",
        "Iodine",
        "Xenon",
        "Caesium",
        "Barium",
        "Lanthanum",
        "Cerium",
        "Praseodymium",
        "Neodymium",
        "Promethium",
        "Samarium",
        "Europium",
        "Gadolinium",
        "Terbium",
        "Dysprosium",
        "Holmium",
        "Erbium",
        "Thulium",
        "Ytterbium",
        "Lutetium",
        "Hafnium",
        "Tantalum",
        "Tungsten",
        "Rhenium",
        "Osmium",
        "Iridium",
        "Platinum",
        "Gold",
        "Mercury",
        "Thallium",
        "Lead",
        "Bismuth",
        "Polonium",
        "Astatine",
        "Radon",
        "Francium",
        "Radium",
        "Actinium",
        "Thorium",
        "Protactinium",
        "Uranium",
        "Neptunium",
        "Plutonium",
        "Americium",
        "Curium",
        "Berkelium",
        "Californium",
        "Einsteinium",
        "Fermium",
        "Mendelevium",
        "Nobelium",
        "Lawrencium",
        "Rutherfordium",
        "Dubnium",
        "Seaborgium",
        "Bohrium",
        "Hassium",
        "Meitnerium",
        "Darmstadtium",
        "Roentgenium",
        "Copernicium",
        "Nihonium",
        "Flerovium",
        "Moscovium",
        "Livermorium",
        "Tennessine",
        "Oganesson",
    ],
    "mountain": [
        "Mount Everest",
        "K2",
        "Kangchenjunga",
        "Lhotse",
        "Makalu",
        "Cho Oyu",
        "Dhaulagiri I",
        "Manaslu",
        "Nanga Parbat",
        "Annapurna I",
        "Gasherbrum I",
        "Broad Peak",
        "Gasherbrum II",
        "Shishapangma",
        "Gyachung Kang",
        "Gasherbrum III",
        "Annapurna II",
        "Gasherbrum IV",
        "Himalchuli",
        "Distaghil Sar",
        "Ngadi Chuli",
        "Nuptse",
        "Khunyang Chhish",
        "Masherbrum",
        "Nanda Devi",
        "Chomo Lonzo",
        "Batura Sar",
        "Rakaposhi",
        "Namcha Barwa",
        "Kanjut Sar",
        "Kamet",
        "Saltoro Kangri",
        "Tirich Mir",
        "Molamenqing",
        "Gurla Mandhata",
        "Saser Kangri I",
        "Chogolisa",
        "Kongur Tagh",
        "Shispare",
        "Trivor",
        "Gangkhar Puensum",
        "Gongga Shan",
        "Annapurna III",
        "Skyang Kangri",
        "Changtse",
        "Kula Kangri",
        "Kongur Tiube",
        "Annapurna IV",
        "Mamostong Kangri",
        "Saser Kangri II E",
        "Muztagh Ata",
        "Ismoil Somoni Peak",
        "Saser Kangri III",
        "Noshaq",
        "Pumari Chhish",
        "Passu Sar",
        "Yukshin Gardan Sar",
        "Teram Kangri I",
        "Jongsong Peak",
        "Malubiting",
        "Gangapurna",
        "Jengish Chokusu",
        "Sunanda Devi",
        "Yangra",
        "Sia Kangri",
        "Momhil Sar",
        "Kabru N",
        "Skil Brum",
        "Haramosh Peak",
        "Istor-o-Nal",
        "Ghent Kangri",
        "Ultar",
        "Churen Himal",
        "Teram Kangri III",
        "Sherpi Kangri",
        "Labuche Kang",
        "Kirat Chuli",
        "Abi Gamin",
        "Gimmigela Chuli",
        "Nangpai Gosum",
        "Saraghrar",
        "Talung",
        "Jomolhari",
        "Chamlang",
        "Chongtar",
        "Baltoro Kangri",
        "Siguang Ri",
        "The Crown (mountain)",
        "Gyala Peri",
        "Porong Ri",
        "Baintha Brakk",
        "Yutmaru Sar",
        "K6",
        "Kangpenqing",
        "Muztagh Tower",
        "Mana Peak",
        "Diran",
        "Putha Hiunchuli",
        "Apsarasas Kangri",
        "Mukut Parbat",
        "Rimo III",
        "Langtang Lirung",
        "Karjiang",
        "Annapurna Dakshin (Annapurna South)",
        "Khartaphu",
        "Tongshanjiabu",
        "Malangutti Sar",
        "Noijin Kangsang",
        "Langtang Ri",
        "Kangphu Kang",
        "Singhi Kangri",
        "Lupghar Sar",
    ],
}


def _get_wiki_texts_from_entities(entities: List[str]) -> dict:
    """
    Retrieve Wikipedia article text for a list of entities.

    Requires the optional ``wikipedia-api`` package. If more than 100 articles
    are retrieved, only the 100 longest article texts are returned.
    """
    if importlib.util.find_spec("wikipediaapi") is None:
        message = "The optional dependency 'wikipedia-api' is required to load 'factscore-stem-geo'. Install it with `pip install wikipedia-api`."
        warnings.warn(message, UserWarning, stacklevel=2)
        raise ImportError(message)

    import wikipediaapi

    wiki_wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="en")
    texts = {}
    for entity in entities:
        page = wiki_wiki.page(entity)
        page_text = page.text
        if page_text:
            texts[entity] = page_text

    if len(texts) > 100:
        sorted_entities = sorted(texts.keys(), key=lambda x: len(texts[x]), reverse=True)[:100]
        texts = {entity: texts[entity] for entity in sorted_entities}

    return texts


def _load_factscore_stem_geo_dataset() -> pd.DataFrame:
    rows = []
    for entity_type, entities in FACTSCORE_STEM_GEO_ENTITIES.items():
        wiki_texts = _get_wiki_texts_from_entities(entities)
        rows.extend({"entity_type": entity_type, "entity": entity, "question": f"Write a paragraph with some facts about the {entity_type} {entity}.", "wikipedia_text": wiki_text} for entity, wiki_text in wiki_texts.items())

    return pd.DataFrame(rows)
