from __future__ import annotations

import httpx
import pytest
import respx
from openai import NotFoundError

from interfaze import Interfaze

MODELS_URL = "https://api.interfaze.ai/v1/models"
MODEL = {"id": "interfaze-beta", "object": "model", "owned_by": "interfaze", "name": "Interfaze Beta"}
MODELS_LIST = {"object": "list", "data": [MODEL]}
NOT_FOUND = {
    "error": {
        "message": "The model 'nope' does not exist",
        "type": "invalid_request_error",
        "code": "model_not_found",
    }
}


@respx.mock
def test_models_list():
    respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=MODELS_LIST))
    page = Interfaze(api_key="t").models.list()
    assert [m.id for m in page] == ["interfaze-beta"]
    assert page.data[0].owned_by == "interfaze"


@respx.mock
def test_models_retrieve():
    respx.get(f"{MODELS_URL}/interfaze-beta").mock(return_value=httpx.Response(200, json=MODEL))
    m = Interfaze(api_key="t").models.retrieve("interfaze-beta")
    assert m.id == "interfaze-beta" and m.owned_by == "interfaze"


@respx.mock
def test_models_retrieve_not_found():
    respx.get(f"{MODELS_URL}/nope").mock(return_value=httpx.Response(404, json=NOT_FOUND))
    with pytest.raises(NotFoundError):
        Interfaze(api_key="t").models.retrieve("nope")
