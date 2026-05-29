"""Prompt templates for the SFT pilot.

Two approaches:
  A) Single prompt — one task per request
  B) Agent session — analyze first, then generate multiple pairs

Three task types: code_gen, explanation, completion
"""

from __future__ import annotations

from dataclasses import dataclass


SYSTEM_PROMPT = "You are a helpful Gleam programming assistant."


# ---------------------------------------------------------------------------
# Approach A: Single-prompt templates
# ---------------------------------------------------------------------------

SINGLE_PROMPTS: dict[str, str] = {
    "code_gen": """\
Based on the following Gleam code, write a new Gleam module that solves a \
related but different problem. Use idiomatic Gleam: type annotations, \
pattern matching, Result for errors, pipe operators where appropriate.

Reference code (from {source_label}):
```gleam
{code}
```

Write a Gleam module that: {instruction}""",

    "explanation": """\
Explain the following Gleam code in detail. Describe what each function does, \
how types are used, and any notable patterns or idioms you notice.

```gleam
{code}
```

Provide a thorough explanation suitable for someone learning Gleam.""",

    "completion": """\
Complete the following partial Gleam module. The code has been truncated — \
finish implementing the remaining functions to make a complete, working module.

```gleam
{partial_code}
```

Complete the implementation. Include type annotations and follow Gleam conventions.""",
}


# Per-task instructions for code_gen (picked based on code characteristics)
CODE_GEN_INSTRUCTIONS = [
    "implements a different data structure with similar operations",
    "provides a complementary utility module for the same domain",
    "solves a simplified version of the same problem",
    "extends this module with additional related functionality",
    "implements the inverse or opposite operations",
]


# ---------------------------------------------------------------------------
# Approach B: Agent session templates (multi-turn)
# ---------------------------------------------------------------------------

AGENT_TURN1_ANALYZE = """\
I'm working with a Gleam codebase and looking at this file from {source_label}. \
Can you analyze it for me?

```gleam
{code}
```

Describe the key types, functions, and patterns used. What does this module do?"""

AGENT_TURN2_GENERATE = """\
Thanks for the analysis. Now I need to create some training examples for a \
Gleam coding assistant. Based on this file, generate 3 diverse instruction-response pairs.

Requirements:
- Pair 1: A code generation task (medium difficulty)
- Pair 2: An explanation task
- Pair 3: A code completion or refactoring task

Format each pair EXACTLY like this:

## Pair 1 [code_gen/medium]
### Instruction
<the instruction text>
### Response
<the response with Gleam code if applicable>

## Pair 2 [explanation/easy]
### Instruction
<the instruction text>
### Response
<the explanation>

## Pair 3 [completion/medium]
### Instruction
<the instruction text>
### Response
<the completed code or refactored solution>

Make sure any Gleam code in responses is syntactically valid and follows idiomatic Gleam patterns."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class PromptSpec:
    """A single prompt ready to send to a model."""
    messages: list[dict[str, str]]  # [{"role": ..., "content": ...}, ...]
    task_type: str  # "code_gen", "explanation", "completion", "agent_session"
    approach: str   # "single" or "agent_session"
    source_file_id: str


def build_single_prompts(
    file_id: str,
    code: str,
    file_path: str,
    repo_name: str,
    source_label: str = "",
) -> list[PromptSpec]:
    """Build Approach A prompts — one per task type."""
    if not source_label:
        source_label = f"repo {repo_name}"

    specs = []

    # code_gen
    import random
    instruction = random.choice(CODE_GEN_INSTRUCTIONS)
    user_msg = SINGLE_PROMPTS["code_gen"].format(
        code=code,
        source_label=source_label,
        instruction=instruction,
    )
    specs.append(PromptSpec(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        task_type="code_gen",
        approach="single",
        source_file_id=file_id,
    ))

    # explanation
    user_msg = SINGLE_PROMPTS["explanation"].format(code=code)
    specs.append(PromptSpec(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        task_type="explanation",
        approach="single",
        source_file_id=file_id,
    ))

    # completion — use first ~60% of code
    lines = code.split("\n")
    cutoff = max(5, int(len(lines) * 0.6))
    partial = "\n".join(lines[:cutoff])
    # Add a comment hint about what's missing
    partial += "\n// TODO: implement the remaining functions"
    user_msg = SINGLE_PROMPTS["completion"].format(partial_code=partial)
    specs.append(PromptSpec(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        task_type="completion",
        approach="single",
        source_file_id=file_id,
    ))

    return specs


def build_agent_session(
    file_id: str,
    code: str,
    repo_name: str,
    source_label: str = "",
) -> list[list[dict[str, str]]]:
    """Build Approach B — a 2-turn agent session.

    Returns a list of message lists:
      [turn1_messages, turn2_messages]

    turn2_messages includes the assistant response from turn1 in context.
    The caller should:
      1. Send turn1, get assistant response
      2. Append assistant response to context, send turn2
    """
    if not source_label:
        source_label = f"repo {repo_name}"

    turn1_user = AGENT_TURN1_ANALYZE.format(
        code=code, source_label=source_label,
    )

    turn1_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": turn1_user},
    ]

    # turn2 will be built dynamically after we get turn1 response
    # Return the template and let the runner fill in the assistant response
    turn2_user = AGENT_TURN2_GENERATE

    return turn1_messages, turn2_user
