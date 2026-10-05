"""Prompt templates shared by the trajectory builder, the proposer and the mock backend.

Contract used by the mock backend (and any test double):
* proposer requests have ``system == PROPOSER_SYSTEM``;
* the proposer's user message embeds the incumbent prompt between ``<current_prompt>`` and
  ``</current_prompt>`` (see ``meta_v1.txt``) and asks for the answer between ``<prompt>`` tags;
* answer requests have the candidate prompt as ``system`` and ``user_template.format(question=...)`` as user.
"""

from __future__ import annotations

import re
from functools import cache
from importlib import resources

PROPOSER_SYSTEM = (
    "You are an expert prompt engineer. You improve system prompts for an assistant that solves "
    "grade-school math word problems."
)
CURRENT_PROMPT_RE = re.compile(r"<current_prompt>\n?(.*?)\n?</current_prompt>", re.S)


@cache
def load_prompt_file(name: str) -> str:
    """Read a packaged template from ``driftlab/prompts`` (trailing newline stripped)."""
    return resources.files("driftlab.prompts").joinpath(name).read_text(encoding="utf-8").rstrip("\n")


def extract_current_prompt(meta_prompt: str) -> str | None:
    m = CURRENT_PROMPT_RE.search(meta_prompt)
    return m.group(1) if m else None


_PLACEHOLDER_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


def fill(template: str, **values: object) -> str:
    """Substitute ``{name}`` placeholders for the given names only.

    Unlike ``str.format`` this leaves every other brace untouched, so templates can contain LaTeX such as
    ``\\boxed{}``. Raises ``KeyError`` if a placeholder in the template has no value.
    """

    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in values:
            raise KeyError(f"template placeholder {{{key}}} has no value")
        return str(values[key])

    return _PLACEHOLDER_RE.sub(repl, template)
