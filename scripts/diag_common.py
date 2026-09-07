"""Shared load path for diagnostic scripts: persist state, do not start loops."""
from __future__ import annotations

import json
from pathlib import Path

from config.settings import config
from core.loop import companion
from core.resource_budgets import refresh_model_pricing


def _state_dir() -> Path:
    return Path(config.data_dir) / "state"


async def load_companion_no_loops():
    """Same restore as ConsciousArchitecture.startup without background tasks."""
    refresh_model_pricing()
    await companion.memory.init()
    state_dir = _state_dir()
    companion.state.load(state_dir / "internal_state.json")
    companion.self_model.load(state_dir / "self_model.json")
    companion.self_model.opinions.load(state_dir / "opinions.json")
    companion.goals.load(state_dir / "goals.json")
    companion.environment.load(state_dir / "environment.json")

    budgets_path = state_dir / "resource_budgets.json"
    if budgets_path.exists():
        bdata = json.loads(budgets_path.read_text())
        companion.api_budget.restore(bdata.get("api", {}))

    life_path = state_dir / "user_life.json"
    if life_path.exists():
        companion.user_life.restore(json.loads(life_path.read_text()))

    theorizer_path = state_dir / "theorizer.json"
    if theorizer_path.exists():
        companion.theorizer.restore(json.loads(theorizer_path.read_text()))

    companion.memory.working.load(state_dir / "working_memory.json")
    await companion.goals.init_unbounded_goals()
    return companion
