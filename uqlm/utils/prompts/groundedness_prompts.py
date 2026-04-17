"""
Prompt templates for the UnifiedGroundednessScorer.

This module provides a single-call prompt that simultaneously decomposes an answer
into atomic claims and verifies each claim against the provided context.
"""


UNIFIED_GROUNDEDNESS_SYSTEM_PROMPT = """\
You are an expert fact-checking assistant specialized in detecting hallucinations in AI-generated text.
Your task is to decompose an answer into atomic factual claims and verify each claim against the provided context.
You must respond with a valid JSON array and nothing else — no markdown fences, no explanation.\
"""


def get_unified_groundedness_prompt(
    context: str,
    answer: str,
    include_reasoning: bool = True,
    include_relevant_context: bool = True,
) -> str:
    """
    Build the user prompt for unified claim decomposition + groundedness verification.

    Parameters
    ----------
    context : str
        The retrieved context (one or more document chunks) used as the ground truth source.
    answer : str
        The generated answer to be analyzed.
    include_reasoning : bool, default=True
        If True, the prompt asks the LLM to provide a short rationale before the verdict.
        Improves accuracy at the cost of longer output.
    include_relevant_context : bool, default=True
        If True, the prompt asks the LLM to quote relevant context excerpts per claim.
        Helps the model ground its verdict in specific evidence.

    Returns
    -------
    str
        The formatted user prompt.
    """
    # Build the per-claim field list dynamically based on flags
    field_lines = [
        '1. "claim" — a single atomic factual statement (subject + verb + object). '
        "Each claim must contain exactly one independent fact.",
        (
            '2. "anchor_text" — the EXACT verbatim contiguous substring from the Answer '
            "that this claim was extracted from. Copy it character-by-character from the Answer. "
            "It must be findable in the Answer via exact string matching."
        ),
    ]
    field_idx = 3

    if include_relevant_context:
        field_lines.append(
            f'{field_idx}. "relevant_context" — list of short verbatim excerpts from the Context '
            "that are relevant to this claim (empty list [] if none found)"
        )
        field_idx += 1

    if include_reasoning:
        field_lines.append(
            f'{field_idx}. "reasoning" — a brief explanation of why you chose the verdict'
        )
        field_idx += 1

    field_lines.append(
        f'{field_idx}. "verdict" — exactly one of: "supported", "baseless", "contradicted"'
    )

    fields_str = "\n".join(field_lines)

    # Build the JSON schema example
    example_fields = '    "claim": "...",\n    "anchor_text": "..."'
    if include_relevant_context:
        example_fields += ',\n    "relevant_context": ["..."]'
    if include_reasoning:
        example_fields += ',\n    "reasoning": "..."'
    example_fields += ',\n    "verdict": "supported"'

    prompt = f"""\
## Context (source of truth)
{context}

## Answer to verify
{answer}

## Task

You must perform TWO steps:

### Step 1: Decompose the Answer into atomic claims

Go through the Answer sentence by sentence. For each sentence, break it into independent atomic \
facts. Each fact should be in the form "subject + verb + object" and contain exactly one piece of \
information. Do NOT use pronouns (he, she, it, they, this, that) — always use the original subject. \
Every factual statement in the Answer must be covered. Do not skip any sentence.

### Step 2: Verify each claim against the Context

For each claim, determine its verdict:

- **"supported"** — the claim is directly entailed or confirmed by the Context.
- **"baseless"** — the claim introduces new information not mentioned or implied anywhere in the Context.
- **"contradicted"** — the claim conflicts with the Context. This includes:
  - Direct factual contradictions (e.g. Context says X, Answer says Y)
  - Distorted or altered details that change the meaning (e.g. wrong dates, wrong numbers, \
wrong names, wrong relationships).
  - Any claim that would give the reader a false understanding of what the Context actually states.

For each claim you MUST provide:
{fields_str}

## Important rules
- Process EVERY sentence in the Answer. Do not skip any part.
- Each sentence should produce at least one claim (unless it contains no factual content).
- "anchor_text" must be copied EXACTLY from the Answer — do not paraphrase, trim, or modify it.
  It must be a contiguous substring that can be located via exact string search in the Answer.

## Output format
Respond with ONLY a JSON array. No markdown, no explanation outside the JSON.

[
  {{
{example_fields}
  }}
]
"""
    return prompt
