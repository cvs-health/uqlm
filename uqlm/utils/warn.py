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

import warnings


class UQLMBetaWarning(Warning):
    """Custom warning class for beta features in UQLM."""

    pass


def beta_warning(message: str):
    """Issues a beta warning with a custom message."""
    warnings.warn(message, category=UQLMBetaWarning, stacklevel=2)


class UQLMDeprecationWarning(Warning):
    """Custom warning class for future deprecation of features in UQLM."""

    pass


def deprecation_warning(message: str):
    """Issues a beta warning with a custom message."""
    warnings.warn(message, category=UQLMDeprecationWarning, stacklevel=2)
