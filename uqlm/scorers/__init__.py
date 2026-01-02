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

from pathlib import Path

from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier
from uqlm.scorers.shortform.baseclass import ShortFormUQ
from uqlm.scorers.shortform import UQEnsemble, SemanticDensity, SemanticEntropy, LLMPanel, WhiteBoxUQ, BlackBoxUQ
from uqlm.scorers.longform.baseclass import LongFormUQ
from uqlm.scorers.longform import LongTextQA, LongTextUQ

__all__ = ["UQEnsemble", "SemanticDensity", "SemanticEntropy", "LLMPanel", "WhiteBoxUQ", "BlackBoxUQ", "LongTextQA", "LongTextUQ", "ShortFormUQ", "LongFormUQ", "UncertaintyQuantifier"]

# Allow submodule imports like `uqlm.scorers.entropy` and `uqlm.scorers.baseclass`
_base_dir = Path(__file__).resolve().parent
for _subdir in ("shortform", "longform"):
    _subpath = _base_dir / _subdir
    if _subpath.exists():
        __path__.append(str(_subpath))

del _base_dir, _subdir, _subpath
