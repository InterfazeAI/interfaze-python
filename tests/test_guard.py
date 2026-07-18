from __future__ import annotations

import pytest
import respx
from conftest import BASIC, last_body, mock_json

from interfaze import Interfaze, InterfazeError
from interfaze._guard import guard_tag


def test_guard_tag_all():
    assert guard_tag(["ALL"]) == "<guard>ALL</guard>"


def test_guard_tag_empty_list_raises():
    with pytest.raises(InterfazeError, match="at least one code"):
        guard_tag([])


def test_guard_tag_invalid_code_raises():
    with pytest.raises(InterfazeError, match="Invalid guard code"):
        guard_tag(["NOT_A_CODE"])


@respx.mock
def test_create_with_all_guard_code():
    route = mock_json(BASIC)
    Interfaze(api_key="t").chat.completions.create(guard=["ALL"], messages=[{"role": "user", "content": "x"}])
    assert "<guard>ALL</guard>" in last_body(route)["messages"][0]["content"]


@respx.mock
def test_create_with_invalid_guard_code_raises_before_request():
    route = mock_json(BASIC)
    with pytest.raises(InterfazeError, match="Invalid guard code"):
        Interfaze(api_key="t").chat.completions.create(
            guard=["NOT_A_CODE"], messages=[{"role": "user", "content": "x"}]
        )
    assert route.calls.call_count == 0
