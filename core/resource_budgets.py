"""Resource budgets — API spend → fatigue. Daily reset at local midnight."""
from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm import Tier

log = logging.getLogger(__name__)

# USD per token. Placeholders — overwritten at startup from OpenRouter's
# /models if reachable (see refresh_model_pricing).
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "x-ai/grok-4.6":                   (5.00 / 1e6, 15.00 / 1e6),
    "google/gemini-3.8-flash":         (0.75 / 1e6,  3.75 / 1e6),
    "google/gemini-3.7-flash":         (0.75 / 1e6,  3.75 / 1e6),
    "z-ai/glm-5.3-flash":              (0.075 / 1e6, 0.25 / 1e6),
    "~deepseek/deepseek-flash-latest": (0.15 / 1e6,  0.60 / 1e6),
    "~typesafe/jev-latest":            (0.042 / 1e6, 0.00),
}
DEFAULT_PRICING = (1.00 / 1e6, 3.00 / 1e6)

_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def _pricing_cache_path() -> Path:
    from config.settings import config
    return Path(config.data_dir) / "state" / "model_pricing.json"


def _apply_pricing_map(data: dict) -> None:
    for mid, prices in data.items():
        if isinstance(prices, (list, tuple)) and len(prices) == 2:
            MODEL_PRICING[str(mid)] = (float(prices[0]), float(prices[1]))


def refresh_model_pricing() -> None:
    """Load cached OpenRouter prices, then refresh from the API if reachable.

    Only the estimation fallback uses this table. The primary path records
    OpenRouter's reported ``usage.cost``.
    """
    from config.settings import config

    cache_path = _pricing_cache_path()
    if cache_path.exists():
        try:
            _apply_pricing_map(json.loads(cache_path.read_text()))
            log.info("Loaded cached model pricing (%d entries)", len(MODEL_PRICING))
        except Exception:
            log.debug("Could not load cached model pricing", exc_info=True)

    headers = {"User-Agent": "Keeper/1.0"}
    if config.openrouter_api_key:
        headers["Authorization"] = f"Bearer {config.openrouter_api_key}"

    try:
        req = urllib.request.Request(_OPENROUTER_MODELS_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode())
        fetched: dict[str, list[float]] = {}
        for item in payload.get("data") or []:
            mid = item.get("id")
            pricing = item.get("pricing") or {}
            prompt = pricing.get("prompt")
            completion = pricing.get("completion")
            if not mid or prompt is None or completion is None:
                continue
            try:
                fetched[mid] = [float(prompt), float(completion)]
            except (TypeError, ValueError):
                continue
        if fetched:
            _apply_pricing_map(fetched)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            serialisable = {k: [v[0], v[1]] for k, v in MODEL_PRICING.items()}
            cache_path.write_text(json.dumps(serialisable, indent=2))
            log.info("Refreshed OpenRouter model pricing (%d models)", len(fetched))
    except Exception:
        log.debug(
            "Could not refresh OpenRouter model pricing; using table/cache",
            exc_info=True,
        )


@dataclass
class APIBudget:
    daily_limit_eur: float = 10.0
    spent_today_eur: float = 0.0
    current_date: str = ""
    spend_by_task: dict[str, float] = field(default_factory=dict)
    spend_by_model: dict[str, float] = field(default_factory=dict)
    calls_today: int = 0
    budget_fatigue_reset: bool = False

    USD_TO_EUR: float = 0.92   # config constant, no live FX

    def __post_init__(self) -> None:
        from config.settings import config
        self.daily_limit_eur = float(config.daily_budget_eur)
        self.USD_TO_EUR = float(config.usd_to_eur)

    def _check_reset(self) -> None:
        today = date.today().isoformat()
        if self.current_date != today:
            self.spent_today_eur = 0.0
            self.spend_by_task = {}
            self.spend_by_model = {}
            self.calls_today = 0
            self.current_date = today
            self.budget_fatigue_reset = True

    def record_cost_usd(self, cost_usd: float, task: str, model: str) -> None:
        self._check_reset()
        cost_eur = float(cost_usd) * self.USD_TO_EUR
        self.spent_today_eur += cost_eur
        self.spend_by_task[task] = self.spend_by_task.get(task, 0.0) + cost_eur
        self.spend_by_model[model] = self.spend_by_model.get(model, 0.0) + cost_eur
        self.calls_today += 1

    def record_estimated(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        task: str,
    ) -> None:
        log.debug(
            "Estimating LLM cost from tokens (no OpenRouter usage.cost) "
            "task=%s model=%s inp=%s out=%s",
            task, model, input_tokens, output_tokens,
        )
        inp_p, out_p = MODEL_PRICING.get(model, DEFAULT_PRICING)
        cost_usd = input_tokens * inp_p + output_tokens * out_p
        self.record_cost_usd(cost_usd, task=task, model=model)

    @property
    def budget_fraction_used(self) -> float:
        """0.0 = nothing spent, 1.0 = budget exhausted."""
        self._check_reset()
        if self.daily_limit_eur <= 0:
            return 1.0
        return min(1.0, self.spent_today_eur / self.daily_limit_eur)

    def compute_fatigue_contribution(self) -> float:
        """
        Maps budget usage to a fatigue value.
        0-50% budget  → 0.0-0.2 fatigue
        50-80% budget → 0.2-0.5 fatigue
        80-100% budget → 0.5-0.9 fatigue
        """
        f = self.budget_fraction_used
        if f < 0.5:
            return f * 0.4
        elif f < 0.8:
            return 0.2 + (f - 0.5) * 1.0
        else:
            return 0.5 + (f - 0.8) * 2.0

    @property
    def remaining_eur(self) -> float:
        self._check_reset()
        return max(0.0, self.daily_limit_eur - self.spent_today_eur)

    def allows(self, tier: "Tier", *, background: bool) -> bool:
        """Gate for LLM work. Conversation (background=False) always proceeds."""
        if not background:
            return True
        f = self.budget_fraction_used
        if f >= 0.85:
            return False
        if f >= 0.60:
            from core.llm import Tier as _Tier
            return tier == _Tier.LOW
        return True

    def to_prompt_context(self) -> str:
        self._check_reset()
        top = sorted(self.spend_by_task.items(), key=lambda kv: -kv[1])[:3]
        top_str = ", ".join(f"{k} €{v:.3f}" for k, v in top) if top else "none"
        return (
            f"API budget: €{self.remaining_eur:.2f} remaining of "
            f"€{self.daily_limit_eur:.2f} "
            f"({self.budget_fraction_used:.0%} used, {self.calls_today} calls today). "
            f"Top tasks: {top_str}."
        )

    def snapshot(self) -> dict:
        return {
            "spent_today_eur": round(self.spent_today_eur, 4),
            "current_date": self.current_date,
            "daily_limit_eur": self.daily_limit_eur,
            "spend_by_task": {k: round(v, 4) for k, v in self.spend_by_task.items()},
            "spend_by_model": {k: round(v, 4) for k, v in self.spend_by_model.items()},
            "calls_today": self.calls_today,
        }

    def restore(self, data: dict) -> None:
        self.spent_today_eur = data.get("spent_today_eur", 0.0)
        self.current_date = data.get("current_date", "")
        self.daily_limit_eur = data.get("daily_limit_eur", self.daily_limit_eur)
        self.spend_by_task = dict(data.get("spend_by_task") or {})
        self.spend_by_model = dict(data.get("spend_by_model") or {})
        self.calls_today = int(data.get("calls_today", 0))
        self._check_reset()
