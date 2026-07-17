from __future__ import annotations

from typing import Any, Dict

JSONSchema = Dict[str, Any]


def empty_task_schema(name: str = "empty_schema") -> Dict[str, Any]:
    """Empty ``response_format`` for raw ``<task>`` runs."""
    return {"type": "json_schema", "json_schema": {"name": name, "schema": {}}}


def response_format(schema: JSONSchema, name: str = "response") -> Dict[str, Any]:
    """Build a structured-output ``response_format`` from a JSON Schema.

    Non-object roots are wrapped in a ``{"result": ...}`` object (structured output requires
    an object root); the wrapped output is then under the ``result`` key.
    """
    return {"type": "json_schema", "json_schema": {"name": name, "schema": _ensure_object_root(schema)}}


def _ensure_object_root(schema: JSONSchema) -> JSONSchema:
    if isinstance(schema, dict) and schema.get("type") == "object":
        return schema
    return {
        "type": "object",
        "properties": {"result": schema},
        "required": ["result"],
        "additionalProperties": False,
    }
