# -*- coding: utf-8 -*-
"""Gemini plugin-contract tests (LLM Provider Plugin plan §6.2 / A.5).

Single-turn tool branches with thought-signature provider_state (A.5
special case), pooled-client injection (D7), usage mapping (Q6), and
neutral→native conversion — no network, real google-genai types.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from google.genai import types
from llm_provider_base import LlmProviderHandler

from llm_provider_gemini import GeminiProvider


class FakePart:
    """Mimics a google-genai Part (attrs + deep model_dump)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self, exclude_none: bool = True):
        def _plain(value):
            if isinstance(value, SimpleNamespace):
                return {k: _plain(v) for k, v in vars(value).items()}
            if isinstance(value, list):
                return [_plain(v) for v in value]
            return value

        return {
            k: _plain(v)
            for k, v in self.__dict__.items()
            if not (exclude_none and v is None)
        }


def _agent(**config_overrides):
    configuration = {"model": "gemini-3-pro"}
    configuration.update(config_overrides)
    return {
        "instructions": "You are terse.",
        "configuration": configuration,
        "llm": {"llm_name": "gemini"},
        "tool_call_role": "tool",
    }


def _ctx(http_client=None, credentials=None):
    return SimpleNamespace(
        http_client=http_client or httpx.Client(),
        credentials=credentials or {"api_key": "sk-gemini"},
        endpoint="https://generativelanguage.googleapis.com",
        model_identifier="gemini-3-pro",
        options={},
        trace_id="tid-1",
        emit=None,
        is_cancelled=None,
        pool_name="httpx_llm_gemini",
    )


def _handler(ctx=None, agent=None):
    return GeminiProvider(
        logging.getLogger("kit"), agent or _agent(), ctx=ctx or _ctx()
    )


def _response(parts, usage=None, finish_reason="STOP"):
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=parts),
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=usage
        or SimpleNamespace(
            promptTokenCount=3, candidatesTokenCount=5, cachedContentTokenCount=2
        ),
        text="".join(getattr(p, "text", "") or "" for p in parts if getattr(p, "text", None)),
    )


# ---------------------------------------------------------------- metadata


def test_handler_is_plugin_base_subclass():
    assert issubclass(GeminiProvider, LlmProviderHandler)


def test_plugin_metadata():
    assert GeminiProvider.api_protocol == "gemini"
    assert GeminiProvider.display_name == "Gemini"
    assert GeminiProvider.default_pool_name == "httpx_llm_gemini"
    md = GeminiProvider.plugin_metadata()
    assert isinstance(md["config_schema"], dict)
    assert "api_key" not in md["config_schema"].get("properties", {})
    assert "api_key" not in md["config_schema"].get("required", [])


def test_init_requires_pooled_client():
    ctx = _ctx()
    ctx.http_client = None  # bypass the helper default
    with pytest.raises(Exception, match="ctx.http_client"):
        GeminiProvider(logging.getLogger("k"), _agent(), ctx=ctx)


# ------------------------------------------------- single-turn handle_response


def test_handle_response_tool_branch_provides_state():
    handler = _handler()
    response = _response(
        [
            FakePart(text="musing", thought=True, thought_signature="c2lnMQ=="),
            FakePart(text="checking."),
            FakePart(
                function_call=SimpleNamespace(
                    id="fc_1", name="find_order", args={"q": "x"}
                )
            ),
        ]
    )
    handler.handle_response(response, [])

    calls = handler.final_output["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["id"] == "fc_1"  # original id for paired replay
    assert json.loads(calls[0]["function"]["arguments"]) == {"q": "x"}
    assert handler.final_output["finish_reason"] == "tool_calls"
    assert handler.final_output["usage"]["total_tokens"] == 8
    assert handler.final_output["usage"]["cache_read_tokens"] == 2
    state = handler.final_output["provider_state"]
    assert state["api_protocol"] == "gemini"
    # thought + text + function_call blocks all serialised (A.5 special case)
    assert len(state["blocks"]) == 3
    assert state["blocks"][2]["function_call"]["id"] == "fc_1"


def test_handle_response_plain_answer_signed_state():
    handler = _handler()
    response = _response(
        [FakePart(text="answer", thought_signature="c2lnOQ==")],
        usage=SimpleNamespace(promptTokenCount=2, candidatesTokenCount=3),
        finish_reason="STOP",
    )
    handler.handle_response(response, [])
    assert handler.final_output["content"] == "answer"
    assert handler.final_output["finish_reason"] == "STOP"
    assert handler.final_output["usage"]["total_tokens"] == 5
    state = handler.final_output["provider_state"]
    assert state is not None
    assert state["blocks"][0]["thought_signature"] == "c2lnOQ=="


def test_handle_response_plain_answer_no_state():
    handler = _handler()
    response = _response([FakePart(text="plain")])
    response.candidates[0].finish_reason = None
    handler.handle_response(response, [])
    assert handler.final_output["provider_state"] is None
    assert handler.final_output["finish_reason"] == "stop"


# ------------------------------------------------- neutral → native (§5.5)


def test_to_native_drops_system_role():
    handler = _handler()
    contents = handler.to_native_messages(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert all(c.role != "system" for c in contents)
    assert contents[-1].role == "user"
    assert contents[-1].parts[0].text == "hi"


def test_to_native_provider_state_replays_parts():
    handler = _handler()
    state = {
        "api_protocol": "gemini",
        "model": "gemini-3-pro",
        "blocks": [
            {"text": "musing", "thought": True, "thought_signature": "c2lnMQ=="},
            {
                "function_call": {
                    "id": "fc_1",
                    "name": "find_order",
                    "args": {"q": 1},
                }
            },
        ],
    }
    contents = handler.to_native_messages(
        [
            {"role": "assistant", "content": "", "provider_state": state},
            {"role": "tool", "tool_call_id": "fc_1", "content": '{"ok": true}', "tool_name": "find_order"},
        ]
    )
    # assistant replay: thought_signature preserved (Gemini 3 requirement)
    replay = contents[0]
    assert replay.parts[0].thought_signature == b"sig1"
    assert replay.parts[1].function_call.id == "fc_1"
    # tool result → FunctionResponse with matching id
    fr = contents[1].parts[0].function_response
    assert fr.id == "fc_1"
    assert fr.name == "find_order"
    assert fr.response == {"ok": True}


def test_to_native_stale_state_dropped():
    handler = _handler()
    stale = {
        "api_protocol": "gemini",
        "model": "some-other-model",
        "blocks": [{"text": "old", "thought_signature": "old"}],
    }
    contents = handler.to_native_messages(
        [{"role": "assistant", "content": "prior", "provider_state": stale}]
    )
    assert any(p.text == "prior" for p in contents[0].parts)


def test_to_native_assistant_tool_calls_converted():
    handler = _handler()
    contents = handler.to_native_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q": 2}'},
                    }
                ],
            }
        ]
    )
    fc = contents[0].parts[0].function_call
    assert fc.id == "call_1"
    assert fc.name == "lookup"
    assert fc.args == {"q": 2}


# ------------------------------------------------------ insert_file (HTTP)


def test_insert_file_routes_through_pooled_client(monkeypatch):
    from unittest import mock

    captured = {}

    def _handler_transport(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, content=b"raw-bytes")

    pooled = httpx.Client(transport=httpx.MockTransport(_handler_transport))
    handler = _handler(ctx=_ctx(http_client=pooled))
    with mock.patch.object(
        handler.client.files,
        "upload",
        return_value=SimpleNamespace(name="files/abc"),
    ):
        result = handler.insert_file(
            file_uri="https://example.com/f.png",
            filename="f.png",
            mime_type="image/png",
        )
    assert captured["url"] == "https://example.com/f.png"
    assert result.name == "files/abc"


def test_real_genai_types_accept_plugin_shapes():
    """Sanity: the plugin's provider_state replay relies on pydantic
    model_validate of raw serialized Parts (google-genai types)."""
    block = {
        "function_call": {"id": "fc_9", "name": "find", "args": {"q": 1}},
        "thought_signature": "c2lnWDM=",
    }
    part = types.Part.model_validate(block)
    assert part.function_call.id == "fc_9"
    assert part.thought_signature == b"sigX3"
