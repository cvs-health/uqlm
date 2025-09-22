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


from rich.progress import SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

HEADERS = ["🤖 Generation", "📈 Scoring", "⚙️ Optimization", "🤖📈 Generation & Scoring", "✂️ Decomposition", "","  - [black]Grading responses against provided ground truth answers with default grader..."]
OPTIMIZATION_TASKS = ["  - [black]Optimizing weights...", "  - [black]Jointly optimizing weights and threshold using grid search...", "  - [black]Optimizing weights using grid search...", "  - [black]Optimizing threshold with grid search..."]


class ConditionalBarColumn(BarColumn):
    def render(self, task):
        if task.description in HEADERS:
            return ""
        return super().render(task)


class ConditionalTimeElapsedColumn(TimeElapsedColumn):
    def render(self, task):
        if task.description in HEADERS:
            return ""
        return super().render(task)


class ConditionalTextColumn(TextColumn):
    def render(self, task):
        if task.description in HEADERS:
            return ""
        elif task.description in OPTIMIZATION_TASKS:
            return f"[progress.percentage]{task.percentage:>3.0f}%"
        return super().render(task)


class ConditionalSpinnerColumn(SpinnerColumn):
    def render(self, task):
        if task.description in HEADERS:
            return ""
        return super().render(task)
