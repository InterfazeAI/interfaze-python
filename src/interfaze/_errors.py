from __future__ import annotations


class InterfazeError(Exception):
    """SDK-level (client-side) error.

    HTTP errors surface as the openai error classes (openai.BadRequestError, etc.),
    which are re-exported from the package root.
    """
