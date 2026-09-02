"""LLM factory — OpenRouter, tiered by task, with universal spend accounting."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

from config.settings import config

log = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class Tier(str, Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


# Task → tier. Every LLM call site in the codebase must appear here.
TASK_TIERS: dict[str, Tier] = {
    # HIGH — user-facing, needs tool-calling fidelity
    "conversation":            Tier.HIGH,

    # MID — structured reasoning, JSON schema output required
    "self_model_analysis":     Tier.MID,
    "hypothesis_generation":   Tier.MID,
    "consistency_check":       Tier.MID,
    "contrarian_review":       Tier.MID,
    "self_theorize":           Tier.MID,
    "future_simulate":         Tier.MID,
    "research_synthesis":      Tier.MID,
    "goal_generation":         Tier.MID,

    # LOW — classification, summarisation, scoring; high volume
    "observation_summary":     Tier.LOW,
    "opinion_detection":       Tier.LOW,
    "opinion_review":          Tier.LOW,
    "pattern_extraction":      Tier.LOW,
    "contradiction_check":     Tier.LOW,
    "query_generation":        Tier.LOW,
    "topic_extraction":        Tier.LOW,
    "action_rating":           Tier.LOW,
    "completion_check":        Tier.LOW,
    "autonomous_pursuit":      Tier.LOW,
}

# Default sampling per task. Preserve the temperatures already in the code.
TASK_PARAMS: dict[str, dict[str, Any]] = {
    "conversation":            {"temperature": 0.7, "max_tokens": 2048},
    "self_model_analysis":     {"temperature": 0.4, "max_tokens": 4096},
    "hypothesis_generation":   {"temperature": 0.4, "max_tokens": 4096},
    "consistency_check":       {"temperature": 0.4, "max_tokens": 2048},
    "contrarian_review":       {"temperature": 0.6, "max_tokens": 2048},
    "self_theorize":           {"temperature": 0.5, "max_tokens": 4096},
    "future_simulate":         {"temperature": 0.6, "max_tokens": 1500},
    "research_synthesis":      {"temperature": 0.5, "max_tokens": 2048},
    "goal_generation":         {"temperature": 0.6, "max_tokens": 2048},
    "observation_summary":     {"temperature": 0.5, "max_tokens": 512},
    "opinion_detection":       {"temperature": 0.3, "max_tokens": 1024},
    "opinion_review":          {"temperature": 0.3, "max_tokens": 1536},
    "pattern_extraction":      {"temperature": 0.3, "max_tokens": 2048},
    "contradiction_check":     {"temperature": 0.3, "max_tokens": 1536},
    "query_generation":        {"temperature": 0.6, "max_tokens": 512},
    "topic_extraction":        {"temperature": 0.6, "max_tokens": 1024},
    "action_rating":           {"temperature": 0.2, "max_tokens": 768},
    "completion_check":        {"temperature": 0.1, "max_tokens": 512},
    "autonomous_pursuit":      {"temperature": 0.7, "max_tokens": 1024},
}


class BudgetCallback(AsyncCallbackHandler):
    """Records OpenRouter's reported cost against the daily budget."""

    def __init__(self, task: str, model: str):
        self.task = task
        self.model = model

    def _record(self, response: LLMResult) -> None:
        from core.loop import companion  # lazy, avoids circular import

        cost_usd = None
        usage: dict[str, Any] = {}

        # OpenRouter returns usage.cost when usage.include is set.
        out = response.llm_output or {}
        usage = out.get("token_usage") or out.get("usage") or {}
        cost_usd = usage.get("cost")

        # Fall back to per-generation metadata on the message itself.
        if cost_usd is None:
            for gen_list in response.generations:
                for gen in gen_list:
                    meta = getattr(gen.message, "response_metadata", {}) or {}
                    m_usage = meta.get("token_usage") or meta.get("usage") or {}
                    if m_usage.get("cost") is not None:
                        cost_usd = m_usage["cost"]
                        usage = m_usage
                        break
                if cost_usd is not None:
                    break

        if cost_usd is not None:
            companion.api_budget.record_cost_usd(
                float(cost_usd), task=self.task, model=self.model
            )
        else:
            # Last resort: estimate from tokens using the pricing table.
            inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            outp = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            if inp or outp:
                companion.api_budget.record_estimated(
                    inp, outp, model=self.model, task=self.task
                )
            else:
                log.warning("No usage data for task=%s model=%s", self.task, self.model)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._record(response)


def model_for(task: str) -> str:
    tier = TASK_TIERS.get(task, Tier.LOW)
    return {
        Tier.LOW: config.model_low,
        Tier.MID: config.model_mid,
        Tier.HIGH: config.model_high,
    }[tier]


def get_llm(task: str, *, json_mode: bool = False, model: str | None = None, **overrides: Any) -> ChatOpenAI:
    """Construct a budget-tracked chat model for a named task.

    Optional ``model=`` bypasses ``model_for(task)`` so conversation can
    degrade HIGH → MID → LOW without a separate task name.
    """
    if task not in TASK_TIERS:
        raise ValueError(f"Unregistered LLM task: {task!r}. Add it to TASK_TIERS.")

    chosen_model = model or model_for(task)
    params = {**TASK_PARAMS.get(task, {}), **overrides}
    params.pop("model", None)

    extra_body: dict[str, Any] = {
        # Report real cost back on every response.
        "usage": {"include": True},
        # Pin the backend so behaviour is reproducible across runs.
        "provider": {"allow_fallbacks": False},
    }
    if json_mode:
        extra_body["response_format"] = {"type": "json_object"}

    # extra_body is a ChatOpenAI constructor kwarg (langchain-openai >= 0.2).
    # Do not drop it — without usage.include the budget falls through to estimation.
    return ChatOpenAI(
        model=chosen_model,
        api_key=config.openrouter_api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/LMotaWiele/Keeper",
            "X-Title": "Keeper",
        },
        extra_body=extra_body,
        callbacks=[BudgetCallback(task=task, model=chosen_model)],
        **params,
    )
