"""APIBudget accounting and gating."""
from __future__ import annotations

from datetime import date, timedelta

from core.llm import Tier
from core.resource_budgets import APIBudget


def test_record_cost_usd_accumulates_and_converts():
    b = APIBudget()
    b.daily_limit_eur = 10.0
    b.USD_TO_EUR = 0.92
    b.current_date = date.today().isoformat()
    b.spent_today_eur = 0.0
    b.spend_by_task = {}
    b.spend_by_model = {}
    b.calls_today = 0

    b.record_cost_usd(1.0, task="conversation", model="x-ai/grok-4.6")
    assert abs(b.spent_today_eur - 0.92) < 1e-9
    assert abs(b.spend_by_task["conversation"] - 0.92) < 1e-9
    assert abs(b.spend_by_model["x-ai/grok-4.6"] - 0.92) < 1e-9
    assert b.calls_today == 1

    b.record_cost_usd(1.0, task="conversation", model="x-ai/grok-4.6")
    assert abs(b.spent_today_eur - 1.84) < 1e-9
    assert b.calls_today == 2


def test_allows_at_fractions():
    b = APIBudget()
    b.daily_limit_eur = 10.0
    b.current_date = date.today().isoformat()

    b.spent_today_eur = 5.0  # 0.5
    assert b.allows(Tier.HIGH, background=True)
    assert b.allows(Tier.MID, background=True)
    assert b.allows(Tier.LOW, background=True)
    assert b.allows(Tier.HIGH, background=False)

    b.spent_today_eur = 7.0  # 0.7
    assert not b.allows(Tier.HIGH, background=True)
    assert not b.allows(Tier.MID, background=True)
    assert b.allows(Tier.LOW, background=True)
    assert b.allows(Tier.HIGH, background=False)

    b.spent_today_eur = 9.0  # 0.9
    assert not b.allows(Tier.HIGH, background=True)
    assert not b.allows(Tier.MID, background=True)
    assert not b.allows(Tier.LOW, background=True)
    assert b.allows(Tier.LOW, background=False)

    b.spent_today_eur = 9.7  # 0.97
    assert not b.allows(Tier.LOW, background=True)
    assert b.allows(Tier.HIGH, background=False)


def test_date_rollover_resets():
    b = APIBudget()
    b.daily_limit_eur = 10.0
    b.USD_TO_EUR = 0.92
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    b.current_date = yesterday
    b.spent_today_eur = 4.0
    b.spend_by_task = {"conversation": 4.0}
    b.spend_by_model = {"x-ai/grok-4.6": 4.0}
    b.calls_today = 7

    _ = b.budget_fraction_used
    assert b.current_date == date.today().isoformat()
    assert b.spent_today_eur == 0.0
    assert b.spend_by_task == {}
    assert b.spend_by_model == {}
    assert b.calls_today == 0


if __name__ == "__main__":
    test_record_cost_usd_accumulates_and_converts()
    print("✓ record_cost_usd accumulates and converts USD→EUR")
    test_allows_at_fractions()
    print("✓ allows() at 0.5 / 0.7 / 0.9 / 0.97")
    test_date_rollover_resets()
    print("✓ date rollover resets spend")
    print("All budget tests passed")
