import os
import json
from typing import List, Dict, Any
import re
import html
import subprocess


def evaluate_python_code(responses: List[str], public_test_cases: List[Any], metadata: List[Any]) -> Dict[str, Any]:
    """
    Evaluates all the Python responses against public test cases.
    """
    results = {"unit_test_passed": [], "stderr": []}
    utils_directory = os.path.join("/".join(os.getcwd().split("/")[:-1]), "uqlm/code")
    for i in range(len(responses)):
        row = {"public_test_cases": public_test_cases[i], "metadata": metadata[i], "response": responses[i]}

        if isinstance(row.get("public_test_cases"), str):
            row["public_test_cases"] = json.loads(row["public_test_cases"])
        if isinstance(row.get("metadata"), str):
            row["metadata"] = json.loads(row["metadata"])
        out = evaluate_row_unified(row, timeout=6, runner_path=os.path.join(utils_directory, "lcb_grader.py"))
        results["unit_test_passed"].append(out.get("unit_test_passed", 0))
        results["stderr"].append(out.get("stderr", ""))
    return results


def evaluate_row_unified(row, timeout=6, runner_path="lcb_grader.py"):
    """
    Evaluates a single row of the dataset using the LCB runner.

    - Sanitizes the model response to isolate valid code.
    - Parses public test cases and determines the testing mode (call-based or stdio).
    - Builds the JSON payload expected by the LCB runner.
    - Invokes the runner in a subprocess, passing the payload to standard input.
    - Captures stdout and stderr from the runner.
    - Decodes the final JSON report produced by the runner.
    - Returns the evaluation results.
    """
    sanitized = sanitize_llm_output(row["response"])

    public_tests = ensure_list_of_dicts(row["public_test_cases"])

    # Detect if row contains a function name → call-based mode
    func_name = None
    if "metadata" in row and isinstance(row["metadata"], dict):
        func_name = row["metadata"].get("func_name")

    # Build payload for LCB runner
    payload = {"code": sanitized, "public_test_cases": public_tests, "timeout": timeout}

    # Only include fn_name if it exists
    if func_name and isinstance(func_name, str) and len(func_name.strip()) > 0:
        payload["fn_name"] = func_name.strip()

    # Call lcb_grader
    res = subprocess.run(["python3", runner_path], input=json.dumps(payload), text=True, capture_output=True)

    # Try to decode LCB output
    try:
        out = json.loads(res.stdout)
    except Exception:
        out = {"unit_test_passed": 0, "results": [], "meta": {"error_code": -4, "error_message": f"Non-JSON stdout: {res.stdout} / stderr: {res.stderr}"}, "stderr": res.stderr, "stdout": res.stdout}

    return out


def sanitize_llm_output(raw: str) -> str:
    """
    Model responses often include extraneous formatting such as markdown fences, explanatory prose, HTML‑escaped characters, and partial or malformed code blocks.

    This function cleans the model response to ensure that only executable Python code is forwarded to the next evaluation stage.
    - Normalizes newline formats and unescapes HTML entities.
    - If the response contains no ``` fences, the raw text is returned after stripping surrounding backticks.
    - If fenced code blocks exist, all blocks are extracted.
    - The longest fenced block is selected (typically the actual code solution).
    - Trailing or malformed backticks are removed.
    """

    if raw is None:
        return ""

    # Normalize newlines and unescape HTML (&gt; -> >)
    text = html.unescape(raw.replace("\r\n", "\n").replace("\r", "\n")).strip()

    #  If pure code was returned (no backticks), return directly
    if "```" not in text:
        # Clean accidental leading/trailing backticks
        return text.strip("`").strip()

    #  Extract fenced blocks (python or generic)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)

    if blocks:
        # Pick the longest block
        code = max(blocks, key=len)
        return code.strip()

    # Remove markdown fences if half-open or malformed
    stripped = re.sub(r"```+", "", text).strip()

    return stripped


def ensure_list_of_dicts(x: str | list) -> list:
    """
    Different dataset rows may express test cases in slightly different formats. To ensure uniformity, this function converts values like `public_test_cases` into proper Python lists, safely handling cases where the value is stored as a JSON string instead of a list.

    Additionally, each row may optionally specify a `func_name`:
    - If provided → the problem is evaluated in call‑based mode.
    - If absent → the problem is evaluated in standard input mode.
    """
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return []
    return x if isinstance(x, list) else []
