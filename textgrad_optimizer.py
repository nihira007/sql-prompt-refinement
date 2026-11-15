"""
textgrad_optimizer.py
=====================

A compact, dependency-free implementation of the TextGrad idea
(Yuksekgonul et al., "TextGrad: Automatic 'Differentiation' via Text") that is
specialised for optimising a text-to-SQL *system prompt*.

Mapping of the deep-learning vocabulary onto this project
---------------------------------------------------------
variable            -> ``PromptVariable``: the system prompt we are optimising
forward pass        -> the LLM turns a question into SQL, the database runs it
loss                -> ``FailureContext.loss_description()``: a natural-language
                       description of what went wrong (SQLite error message, or
                       how the result differs from the expected one)
backward pass       -> ``TextGradOptimizer.compute_gradient()``: a critic LLM
                       explains how the *prompt* should change -> textual gradient
optimizer step      -> ``TextGradOptimizer.propose_update()``: an LLM applies the
                       gradients to the prompt (textual gradient descent, TGD)

The step is split into *propose* and *apply* on purpose: the pipeline measures
the candidate prompt before committing to it, which turns plain TGD into a
validation-gated variant that cannot silently make the prompt worse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from llm_client import BaseLLM

PROMPT_ROLE = (
    "the system prompt that instructs an LLM to translate natural-language questions into correct "
    "SQLite queries for a given schema"
)

# --------------------------------------------------------------------------- #
# Prompt templates for the two LLM-driven TextGrad operations
# --------------------------------------------------------------------------- #

GRADIENT_SYSTEM = """\
You are a meticulous prompt-debugging critic acting as the backward pass of a TextGrad optimisation loop.

An LLM received a SYSTEM PROMPT (the variable being optimised) and was asked to translate a question into a \
SQLite query. The outcome was bad: the query failed, returned a wrong result, or returned the right result \
through an inefficient query plan. Produce a textual gradient: concrete, constructive feedback on how the \
SYSTEM PROMPT should change so that a model following it avoids this kind of mistake.

Rules for the feedback:
- Criticise and improve the PROMPT, not the individual SQL query.
- Diagnose the root cause from the error message, from the difference between the actual and expected results, \
or, for inefficient queries, from the query plan (full scans, correlated subqueries re-run per row, temporary \
structures) and the measured cost.
- Recommend general, reusable guidance. Never hard-code literal values, table contents, or the answer to this \
particular question.
- Be specific enough to act on. Write 2-4 sentences. Do not rewrite the whole prompt."""

OPTIMIZER_SYSTEM = """\
You are the optimiser step ("textual gradient descent") of a TextGrad loop. You receive the current value of a \
SYSTEM PROMPT variable plus feedback (textual gradients) gathered from queries that failed. Produce an improved \
version of the prompt.

Requirements:
- Keep everything in the prompt that still helps and fold the feedback in as general, reusable guidelines.
- Do NOT include the database schema (it is supplied separately), specific question text, or hard-coded answers.
- Stay concise: the prompt must remain under {max_chars} characters. Merge or drop redundant rules instead of \
only appending.
- The prompt must keep telling the model to return a single SQLite SELECT query and nothing else.
- Reply with the complete new prompt between <IMPROVED_PROMPT> and </IMPROVED_PROMPT> tags and nothing else."""


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class PromptVariable:
    """The optimisable text (analogue of ``tg.Variable(requires_grad=True)``)."""

    value: str
    role_description: str = PROMPT_ROLE
    requires_grad: bool = True
    gradients: list[str] = field(default_factory=list)
    versions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.value = self.value.strip()
        self.versions = [self.value]

    def zero_grad(self) -> None:
        self.gradients.clear()

    def update(self, new_value: str) -> None:
        self.value = new_value.strip()
        self.versions.append(self.value)


@dataclass
class FailureContext:
    """Everything the critic needs to know about one failed forward pass."""

    question: str
    sql: str
    failure_type: str  # "execution_error" | "result_mismatch" | "inefficient_plan"
    error: str | None = None  # database error text
    mismatch_reason: str = ""  # why the result differs from the expected one
    actual_preview: str = ""
    expected_preview: str = ""
    reference_sql: str | None = None  # gold SQL of a *training* question, if any
    plan_text: str = ""  # EXPLAIN QUERY PLAN of the generated query
    plan_issues: list[str] = field(default_factory=list)
    actual_steps: int = 0  # SQLite VM steps of the generated query
    expected_steps: int = 0  # VM steps of the reference query (0 if unknown)

    def loss_description(self) -> str:
        """The "loss": a natural-language description of the failure."""
        if self.failure_type == "execution_error":
            return f"The query failed to execute. Database error: {self.error}"
        if self.failure_type == "inefficient_plan":
            cost = f"it needed about {self.actual_steps} SQLite VM steps"
            if self.expected_steps:
                cost += (
                    f" versus about {self.expected_steps} for an efficient reference "
                    f"({self.actual_steps / max(self.expected_steps, 1):.1f}x more work)"
                )
            parts = [f"The query returned the correct result but is inefficient: {cost}."]
            if self.plan_text:
                parts.append(f"SQLite query plan:\n{self.plan_text}")
            if self.plan_issues:
                parts.append("Detected problems:\n" + "\n".join(f"- {i}" for i in self.plan_issues))
            return "\n".join(parts)
        parts = [f"The query executed but returned a wrong result: {self.mismatch_reason}."]
        if self.actual_preview:
            parts.append(f"Actual result:\n{self.actual_preview}")
        if self.expected_preview:
            parts.append(f"Expected result:\n{self.expected_preview}")
        return "\n".join(parts)


@dataclass
class UpdateProposal:
    """Candidate prompt produced by one optimiser step (not yet applied)."""

    new_value: str | None
    raw_response: str
    valid: bool
    changed: bool
    reason: str


def extract_improved_prompt(text: str) -> str | None:
    """Pull the prompt out of <IMPROVED_PROMPT>...</IMPROVED_PROMPT> (None if missing)."""
    match = re.search(r"<IMPROVED_PROMPT>\s*(.*?)\s*</IMPROVED_PROMPT>", text or "", re.DOTALL)
    return match.group(1).strip() if match and match.group(1).strip() else None


# --------------------------------------------------------------------------- #
# The optimiser
# --------------------------------------------------------------------------- #


class TextGradOptimizer:
    """Backward pass (textual gradients) and TGD step for a prompt variable."""

    def __init__(self, llm: BaseLLM, max_prompt_chars: int = 3500) -> None:
        self.llm = llm
        self.max_prompt_chars = max_prompt_chars

    # -- backward -------------------------------------------------------- #
    def compute_gradient(self, variable: PromptVariable, ctx: FailureContext, schema: str) -> str:
        """
        Ask the critic LLM how ``variable`` should change given one failure.
        The feedback is stored in ``variable.gradients`` and also returned.
        """
        reference = ""
        if ctx.reference_sql:
            reference = (
                "<REFERENCE_SQL>\n"
                f"{ctx.reference_sql}\n"
                "</REFERENCE_SQL>\n"
                "(A reference solution to this training question is given only to help your diagnosis. "
                "Extract the general lesson; do not copy question-specific details into the feedback.)\n"
            )
        user = (
            f"<VARIABLE_ROLE>\n{variable.role_description}\n</VARIABLE_ROLE>\n"
            f"<PROMPT>\n{variable.value}\n</PROMPT>\n"
            f"<SCHEMA>\n{schema}\n</SCHEMA>\n"
            f"<QUESTION>\n{ctx.question}\n</QUESTION>\n"
            f"<GENERATED_SQL>\n{ctx.sql}\n</GENERATED_SQL>\n"
            f"<LOSS>\n{ctx.loss_description()}\n</LOSS>\n"
            f"{reference}"
            "Write the feedback for the SYSTEM PROMPT now."
        )
        gradient = self.llm.invoke(GRADIENT_SYSTEM, user, role="gradient").strip()
        variable.gradients.append(gradient)
        return gradient

    # -- optimiser step -------------------------------------------------- #
    def propose_update(self, variable: PromptVariable) -> UpdateProposal:
        """
        Apply all gradients collected on ``variable`` (one mini-batch) in a single
        TGD step and return the candidate prompt. The variable is NOT modified.
        """
        if not variable.gradients:
            return UpdateProposal(None, "", False, False, "no gradients to apply")

        feedback = "\n\n".join(f"[Feedback {i}] {g}" for i, g in enumerate(variable.gradients, start=1))
        user = (
            f"<VARIABLE_ROLE>\n{variable.role_description}\n</VARIABLE_ROLE>\n"
            f"<CURRENT_PROMPT>\n{variable.value}\n</CURRENT_PROMPT>\n"
            f"<FEEDBACK>\n{feedback}\n</FEEDBACK>\n"
            "Write the improved prompt now."
        )
        system = OPTIMIZER_SYSTEM.format(max_chars=self.max_prompt_chars)
        raw = self.llm.invoke(system, user, role="optimize")

        candidate = extract_improved_prompt(raw)
        if candidate is None:
            return UpdateProposal(None, raw, False, False, "optimiser reply had no <IMPROVED_PROMPT> block")
        if len(candidate) > self.max_prompt_chars:
            return UpdateProposal(
                None, raw, False, False,
                f"candidate prompt too long ({len(candidate)} > {self.max_prompt_chars} characters)",
            )
        if candidate == variable.value:
            return UpdateProposal(candidate, raw, True, False, "optimiser left the prompt unchanged")
        return UpdateProposal(candidate, raw, True, True, "candidate prompt produced")
