"""
Resource budgets — tie internal drives to real resource constraints.

Curiosity controls search willingness. Fatigue tracks API spend.
Both reset daily at midnight UTC.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class SearchBudget:
    """Daily Tavily search quota, gated by curiosity level."""

    base_daily_quota: int = 20
    max_daily_quota: int = 60
    searches_today: int = 0
    current_date: str = ""

    def _check_reset(self):
        today = date.today().isoformat()
        if self.current_date != today:
            self.searches_today = 0
            self.current_date = today

    def effective_quota(self, curiosity: float) -> int:
        """
        Current quota based on curiosity level.
        curiosity=0.0 → base_daily_quota
        curiosity=1.0 → max_daily_quota
        """
        return int(
            self.base_daily_quota
            + (self.max_daily_quota - self.base_daily_quota) * curiosity
        )

    def can_search(self, curiosity: float) -> bool:
        self._check_reset()
        return self.searches_today < self.effective_quota(curiosity)

    def record_search(self):
        self._check_reset()
        self.searches_today += 1

    @property
    def remaining(self) -> int:
        """Remaining at max curiosity (upper bound)."""
        self._check_reset()
        return max(0, self.max_daily_quota - self.searches_today)

    def to_prompt_context(self, curiosity: float) -> str:
        self._check_reset()
        quota = self.effective_quota(curiosity)
        remaining = max(0, quota - self.searches_today)
        return (
            f"Search budget: {remaining}/{quota} remaining today "
            f"(curiosity {curiosity:.2f} → quota {quota})"
        )

    def snapshot(self) -> dict:
        return {
            "searches_today": self.searches_today,
            "current_date": self.current_date,
            "base_daily_quota": self.base_daily_quota,
            "max_daily_quota": self.max_daily_quota,
        }

    def restore(self, data: dict):
        self.searches_today = data.get("searches_today", 0)
        self.current_date = data.get("current_date", "")
        self.base_daily_quota = data.get("base_daily_quota", 20)
        self.max_daily_quota = data.get("max_daily_quota", 60)


@dataclass
class APIBudget:
    """
    Daily Claude API budget tracker.
    Fatigue rises as spend approaches the daily limit.
    """

    daily_limit_eur: float = 10.0
    spent_today_eur: float = 0.0
    current_date: str = ""

    # Anthropic pricing (update as needed)
    # Claude 3.5 Sonnet: $3/M input, $15/M output
    INPUT_COST_PER_TOKEN: float = 3.0 / 1_000_000
    OUTPUT_COST_PER_TOKEN: float = 15.0 / 1_000_000
    USD_TO_EUR: float = 0.92

    def _check_reset(self):
        today = date.today().isoformat()
        if self.current_date != today:
            self.spent_today_eur = 0.0
            self.current_date = today

    def record_usage(self, input_tokens: int, output_tokens: int):
        self._check_reset()
        cost_usd = (
            input_tokens * self.INPUT_COST_PER_TOKEN
            + output_tokens * self.OUTPUT_COST_PER_TOKEN
        )
        self.spent_today_eur += cost_usd * self.USD_TO_EUR

    @property
    def budget_fraction_used(self) -> float:
        """0.0 = nothing spent, 1.0 = budget exhausted."""
        self._check_reset()
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

    def can_afford(self, estimated_tokens: int = 2000) -> bool:
        """Quick check: is there budget for a typical LLM call?"""
        self._check_reset()
        est_cost = estimated_tokens * self.OUTPUT_COST_PER_TOKEN * self.USD_TO_EUR
        return self.spent_today_eur + est_cost < self.daily_limit_eur

    def to_prompt_context(self) -> str:
        self._check_reset()
        return (
            f"API budget: €{self.remaining_eur:.2f} remaining of "
            f"€{self.daily_limit_eur:.2f} daily "
            f"({self.budget_fraction_used:.0%} used → "
            f"fatigue contribution: {self.compute_fatigue_contribution():.2f})"
        )

    def snapshot(self) -> dict:
        return {
            "spent_today_eur": round(self.spent_today_eur, 4),
            "current_date": self.current_date,
            "daily_limit_eur": self.daily_limit_eur,
        }

    def restore(self, data: dict):
        self.spent_today_eur = data.get("spent_today_eur", 0.0)
        self.current_date = data.get("current_date", "")
        self.daily_limit_eur = data.get("daily_limit_eur", 10.0)
