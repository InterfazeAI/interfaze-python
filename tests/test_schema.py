from __future__ import annotations

import respx
from conftest import TASK_OCR, last_body, mock_json

from interfaze import Interfaze, empty_task_schema, response_format
from interfaze._schema import _ensure_object_root


def test_empty_task_schema_default_name():
    schema = empty_task_schema()
    assert schema == {"type": "json_schema", "json_schema": {"name": "empty_schema", "schema": {}}}


def test_empty_task_schema_custom_name():
    schema = empty_task_schema(name="custom")
    assert schema["json_schema"]["name"] == "custom"


def test_response_format_object_root_passthrough():
    obj_schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    rf = response_format(obj_schema, name="my_schema")
    assert rf == {"type": "json_schema", "json_schema": {"name": "my_schema", "schema": obj_schema}}


def test_response_format_non_object_root_wrapped():
    array_schema = {"type": "array", "items": {"type": "string"}}
    rf = response_format(array_schema)
    schema = rf["json_schema"]["schema"]
    assert schema == {
        "type": "object",
        "properties": {"result": array_schema},
        "required": ["result"],
        "additionalProperties": False,
    }


def test_ensure_object_root_scalar_root():
    assert _ensure_object_root({"type": "string"}) == {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
        "additionalProperties": False,
    }


def test_ensure_object_root_object_passthrough():
    obj = {"type": "object", "properties": {}}
    assert _ensure_object_root(obj) is obj


@respx.mock
def test_task_plus_empty_properties_schema_does_not_raise():
    """`_is_non_empty_schema` treats an empty `properties` dict as no schema at all, so
    combining it with `task` is allowed — unlike a schema with real properties."""
    route = mock_json(TASK_OCR)
    Interfaze(api_key="t").chat.completions.create(
        task="ocr",
        messages=[{"role": "user", "content": "x"}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "s", "schema": {"type": "object", "properties": {}}},
        },
    )
    assert last_body(route)["response_format"] == empty_task_schema()
