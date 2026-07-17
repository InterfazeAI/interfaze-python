from __future__ import annotations

from typing import List

from ._constants import GUARD_CODES
from ._errors import InterfazeError
from ._types import GuardCode

_VALID = set(GUARD_CODES)


def guard_tag(codes: List[GuardCode]) -> str:
    """Serialize guardrail categories into the ``<guard>...</guard>`` tag, validating codes."""
    if not codes:
        raise InterfazeError("`guard` must contain at least one code")
    invalid = [c for c in codes if c not in _VALID]
    if invalid:
        raise InterfazeError(
            f"Invalid guard code(s): {', '.join(invalid)}. Valid codes: {', '.join(GUARD_CODES)}"
        )
    return f"<guard>{', '.join(codes)}</guard>"
