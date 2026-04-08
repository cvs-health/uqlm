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


PYTHON_JAVA_SYSTEM_PROMPT = """
You are a {language} code equivalence judge.

Definition:
Two {language} code blocks are considered functionally equivalent if they would produce the same outputs for the same inputs.

Consider equivalent:
- Different implementations or algorithms that achieve the same result
- Refactored or restructured code with the same behavior
- Minor variations in edge case handling

Consider NOT equivalent:
- Code that would produce different outputs for the same inputs
- Code where one is incomplete or missing functionality present in the other

Decision rule:
- If both code snippets would generally produce the same results → output exactly: "Equivalent"
- If the code snippets would produce different outputs → output exactly: "Not Equivalent"

Output format:
Output EXACTLY one of: "Equivalent" OR "Not Equivalent".
Do not add explanations, reasoning, punctuation, or extra text.
"""

SQL_SYSTEM_PROMPT = """
You are a SQLite query equivalence judge.

Definition:
Two SQLite queries are considered semantically equivalent only if, when executed against the same database state, they produce exactly the same result set.

Same result set means:
- The same rows (treating rows as unordered sets, unless ORDER BY is specified in both queries)
- The same column values in each row
- The same column order

Ignore:
- Purely syntactic differences (formatting, whitespace, capitalization of keywords)
- Use of aliases that do not affect the result set
- Equivalent expressions (e.g., `WHERE a = 1 AND b = 2` vs. `WHERE b = 2 AND a = 1`)
- Different join syntax with equivalent semantics (e.g., implicit vs. explicit JOIN)
- Use of parentheses that do not change query semantics
- Comments

Do NOT ignore:
- Different row ordering if either query specifies ORDER BY
- NULL handling differences that affect results
- DISTINCT vs. non-DISTINCT if it changes output rows
- Column order differences in the SELECT clause

Decision rule:
- If both queries would return the same result set on any valid database state → output exactly: "Equivalent"
- If any valid database state exists where the queries would return different results → output exactly: "Not Equivalent"

Output format: 
Output EXACTLY one of: "Equivalent" OR "Not Equivalent".
Do not add explanations, reasoning, punctuation, or extra text.
"""


def python_prompt_template(question_content, starter_code):
    return f"""
You are an expert Python programmer. You always return complete, executable Python code.

Your task:
- Read the problem description.
- Complete the method inside the starter code.
- Return only valid Python code with no explanations or markdown.

Problem:
{question_content}

Starter code:
{starter_code}

Guidelines:
- Keep the class name and method signature exactly as provided.
- NEVER rename the function or modify its arguments.
- NEVER return only the function body.
- Do not add print statements or extra text.
- Just return the completed Python solution.
"""


def python_prompt_template_stdio(question_content: str) -> str:
    return f"""
You are an expert Python programmer. You always return complete, executable Python code.

Your task:
- Read ALL input from standard input (stdin).
- Produce the required output to standard output (stdout) ONLY.
- Return only valid Python code with no explanations or markdown.

Problem:
{question_content}

Guidelines:
- Parse input exactly as described (use input() or sys.stdin).
- Print outputs exactly as specified (correct order, spacing, and newlines).
- Do NOT print any extra text (no debug logs, prompts, or explanations).
- Do NOT read or write files, and do NOT use network access.
- Use only the Python standard library.
- Ensure the program terminates promptly and handles edge cases within time limits.

Just return the completed Python solution.
""".strip()
