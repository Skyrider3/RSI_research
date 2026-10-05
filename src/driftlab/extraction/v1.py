"""Extractor v1 (strict, FROZEN): only a plain number inside the last ``\\boxed{}`` or after ``####``.

1. ``select_marker`` (shared with v2): the later of the last balanced box and the last QUALIFYING ``####``
   line (the rest of the line starts like a number; headings such as ``#### Step 4: Verify`` and empty
   ``####`` lines are skipped, so ``\\boxed{72}`` followed by such a heading still yields 72).
2. Payload: the stripped box content, or the ``####`` line with ONE sentence-final '.' removed.
3. Accept only ``-?\\d+(?:\\.\\d+)?``; anything else gives ``None`` (``#### $42``, ``#### 1,234``,
   ``#### 42 apples``, ``#### .5``). No fallback to the other marker.
"""

from __future__ import annotations

import re

from driftlab.extraction.common import (
    NO_EXTRACTION,
    Extraction,
    marker_payload,
    select_marker,
    to_fraction,
)

STRICT_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract(text: str) -> Extraction:
    """Strict extraction; a rejected marker is reported as ``method="none"`` with its content and span."""
    marker = select_marker(text)
    if marker is None:
        return NO_EXTRACTION
    kind, content, span = marker
    content, span = marker_payload(kind, content, span)
    if STRICT_NUMBER_RE.fullmatch(content):
        value = to_fraction(content)
        if value is not None:
            return Extraction(value, kind, span, content)
    return Extraction(None, "none", span, content)
