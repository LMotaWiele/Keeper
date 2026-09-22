"""Routing v2 contract tests — factory knobs and Jev adapter (no network)."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from config.settings import config
from core.jev import (
    JEV_SPECS,
    JevAnswer,
    build_payload,
    noul_allows,
    parse_answers,
    slice_state,
    validate_questions,
    verdict_from_noul,
)
from core.llm import TASK_TIERS, Tier, fallback_model_for, get_llm, model_for


class _FakeChat:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model")
        self._fallbacks = None

    def with_fallbacks(self, others):
        self._fallbacks = others
        return self

    def bind_tools(self, tools):
        self._tools = tools
        return self


def test_action_bias_text_is_low():
    assert TASK_TIERS["action_bias_text"] is Tier.LOW
    assert TASK_TIERS["action_bias_eval"] is Tier.LOW
    assert TASK_TIERS["conversation"] is Tier.HIGH


def test_default_and_fallback_slugs():
    assert config.model_high == "google/gemini-3.8-flash"
    assert config.model_mid == "google/gemini-3.8-flash"
    assert config.model_low == "z-ai/glm-5.3-flash"
    assert fallback_model_for("conversation", config.model_high) == config.model_high_fallback
    assert fallback_model_for("opinion_detection", config.model_low) == config.model_low_fallback
    assert fallback_model_for("self_model_analysis", config.model_mid) == config.model_mid_fallback
    assert fallback_model_for("conversation", config.model_high_fallback) is None


def test_low_glm_uses_effort_low_not_off():
    with patch("core.llm.ChatOpenAI", _FakeChat):
        llm = get_llm("opinion_detection", wrap_fallback=False)
    extra = llm.kwargs["extra_body"]
    assert extra["reasoning"] == {"effort": "low"}
    assert extra["provider"]["sort"] == "price"
    assert extra["provider"]["allow_fallbacks"] is False
    assert extra["usage"]["include"] is True


def test_low_deepseek_fallback_disables_reasoning():
    created = []

    class Capture(_FakeChat):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(self)

    with patch("core.llm.ChatOpenAI", Capture):
        get_llm("opinion_detection")
    assert created[1].model == config.model_low_fallback
    extra = created[1].kwargs["extra_body"]
    assert extra["reasoning"] == {"enabled": False, "effort": "none"}


def test_high_conversation_exacto_no_reasoning_override():
    with patch("core.llm.ChatOpenAI", _FakeChat):
        llm = get_llm("conversation", wrap_fallback=False)
    extra = llm.kwargs["extra_body"]
    assert extra["provider"]["sort"] == "exacto"
    assert "reasoning" not in extra


def test_mid_task_price_sort_even_when_slug_matches_high():
    with patch("core.llm.ChatOpenAI", _FakeChat):
        llm = get_llm("self_model_analysis", wrap_fallback=False)
    extra = llm.kwargs["extra_body"]
    assert extra["provider"]["sort"] == "price"
    assert "reasoning" not in extra


def test_degraded_conversation_low_turns_reasoning_off():
    with patch("core.llm.ChatOpenAI", _FakeChat):
        llm = get_llm("conversation", model=config.model_low, wrap_fallback=False)
    extra = llm.kwargs["extra_body"]
    assert extra["provider"]["sort"] == "price"
    # GLM cannot disable reasoning; cheapest accepted knob is effort=low.
    assert extra["reasoning"]["effort"] == "low"


def test_get_llm_wraps_named_slug_fallback():
    created = []

    class Capture(_FakeChat):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(self)

    with patch("core.llm.ChatOpenAI", Capture):
        llm = get_llm("opinion_detection")
    assert len(created) == 2
    assert created[0].model == config.model_low
    assert created[1].model == config.model_low_fallback
    assert llm._fallbacks == [created[1]]


def test_wrap_fallback_false_is_primary_only():
    created = []

    class Capture(_FakeChat):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(self)

    with patch("core.llm.ChatOpenAI", Capture):
        get_llm("opinion_detection", wrap_fallback=False)
    assert len(created) == 1


def test_force_answer_uses_turn_tier():
    seen = {}

    class FakeLLM:
        async def ainvoke(self, messages):
            class R:
                content = "forced"
            return R()

    def fake_get_llm(task, **kwargs):
        seen["task"] = task
        seen["model"] = kwargs.get("model")
        return FakeLLM()

    async def _run():
        with patch("agent.graph.get_llm", fake_get_llm):
            from agent.graph import force_answer
            result = await force_answer({
                "user_id": 1,
                "messages": [],
                "system_prompt": "sys",
                "tool_calls_pending": False,
                "response_text": "",
                "tool_calls_made": 16,
                "tier": "low",
                "forced_answer": False,
            })
        return result

    result = asyncio.run(_run())
    assert seen["task"] == "conversation"
    assert seen["model"] == config.model_low
    assert result["forced_answer"] is True
    assert result["response_text"] == "forced"


def test_validate_questions_rejects_bad_payloads():
    validate_questions({"a": dict(JEV_SPECS["stance_present"])})
    try:
        validate_questions({})
        assert False, "empty questions should fail"
    except ValueError:
        pass
    try:
        validate_questions({"a": {"type": "choice", "criteria": {"only": "one"}}})
        assert False, "choice needs two options"
    except ValueError:
        pass
    try:
        validate_questions({"a": {"type": "score", "criteria": ["one"]}})
        assert False, "score needs two levels"
    except ValueError:
        pass
    try:
        validate_questions({"a": {"type": "prose"}})
        assert False, "prose is not a primitive"
    except ValueError:
        pass


def test_build_payload_does_not_include_mid_options():
    payload = build_payload(
        {"user_text": "hello"},
        {"stance_present": dict(JEV_SPECS["stance_present"])},
        config.jev_model,
    )
    dumped = json.dumps(payload)
    assert "elo" not in dumped.lower()
    assert payload["questions"]["stance_present"]["type"] == "noul"
    assert payload["model"] == config.jev_model


def test_slice_state_caps_at_hard_limit():
    from core.jev import HARD_SLICE_CHARS
    huge = "x" * (HARD_SLICE_CHARS + 500)
    sliced = slice_state(huge)
    assert isinstance(sliced, str)
    assert len(sliced) == HARD_SLICE_CHARS
    small = {"ok": True}
    assert slice_state(small) == small


def test_parse_answers_and_noul_branching():
    parsed = parse_answers({
        "answers": {
            "stance_present": {"type": "noul", "noul": 0.2},
            "worth_rich_observation": {"type": "noul", "noul": 0.8},
        }
    })
    assert noul_allows(parsed["stance_present"], 0.45) is False
    assert noul_allows(parsed["worth_rich_observation"], 0.50) is True
    assert noul_allows(None, 0.45) is True
    assert noul_allows(JevAnswer(), 0.45) is True
    low_conf = JevAnswer(noul=0.1, confidence=0.2)
    assert noul_allows(low_conf, 0.45) is True


def test_verdict_from_noul_mapping():
    assert verdict_from_noul(0.70)[0] == "HELD"
    assert verdict_from_noul(0.20)[0] == "VIOLATED"
    assert verdict_from_noul(0.50)[0] == "INVALID"
    assert verdict_from_noul(0.65)[0] == "HELD"
    assert verdict_from_noul(0.35)[0] == "VIOLATED"
    v, reason = verdict_from_noul(0.91)
    assert reason.startswith("jev noul=")


def test_jev_decide_retries_503_then_fail_open():
    import os
    os.environ["JEV_ALLOW_IN_TESTS"] = "1"
    calls = {"n": 0}

    class FakeResp:
        def __init__(self, status, payload=None, text="err"):
            self.status_code = status
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            calls["n"] += 1
            return FakeResp(503, text="unavailable")

    class FakeBudget:
        def allows(self, tier, *, background):
            return True

    async def _run():
        with patch("core.jev.httpx.AsyncClient", FakeClient), \
             patch("core.loop.companion") as companion:
            companion.api_budget = FakeBudget()
            from core.jev import jev_decide
            return await jev_decide(
                {"x": 1},
                {"stance_present": dict(JEV_SPECS["stance_present"])},
                background=False,
                task="post_turn_gates",
            )

    try:
        result = asyncio.run(_run())
        assert result is None
        assert calls["n"] == 2
    finally:
        os.environ.pop("JEV_ALLOW_IN_TESTS", None)


def test_jev_decide_parses_success():
    import os
    os.environ["JEV_ALLOW_IN_TESTS"] = "1"

    class FakeResp:
        status_code = 200
        text = "ok"

        def json(self):
            return {
                "answers": {"stance_present": {"noul": 0.9}},
                "usage": {"prompt_tokens": 12, "cost": 0.0},
            }

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return FakeResp()

    class FakeBudget:
        def allows(self, tier, *, background):
            return True

        def record_cost_usd(self, *a, **k):
            return None

        def record_estimated(self, *a, **k):
            return None

    async def _run():
        with patch("core.jev.httpx.AsyncClient", FakeClient), \
             patch("core.jev._in_pytest", return_value=False), \
             patch("core.loop.companion") as companion:
            companion.api_budget = FakeBudget()
            from core.jev import jev_decide
            return await jev_decide(
                {"user_text": "hi"},
                {"stance_present": dict(JEV_SPECS["stance_present"])},
                background=False,
                task="post_turn_gates",
            )

    try:
        result = asyncio.run(_run())
        assert result is not None
        assert abs(result["stance_present"].noul - 0.9) < 1e-9
    finally:
        os.environ.pop("JEV_ALLOW_IN_TESTS", None)


def test_model_for_uses_task_tiers():
    assert model_for("conversation") == config.model_high
    assert model_for("opinion_detection") == config.model_low
    assert model_for("self_model_analysis") == config.model_mid
    assert model_for("action_bias_text") == config.model_low


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"✓ {fn.__name__}")
    print(f"All {len(tests)} routing v2 tests passed")
