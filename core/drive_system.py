"""
Drive system — internal motivational hungers.

Drives are persistent urges that build up over time and get satisfied
(temporarily) by specific kinds of activity. They're the bridge between
internal state and the goal system — drives create motivation,
goals channel it into action.

Unlike emotions (arousal, valence) which react to what's happening,
drives accumulate in the background. An unresolved question keeps
nagging. An unexplored topic keeps pulling. A long silence builds
the urge to reach out.

Each drive has:
  - intensity: how strong the urge is right now (0–1)
  - buildup_rate: how fast it accumulates when unsatisfied
  - decay_rate: how fast it fades after being satisfied
  - last_satisfied: when it was last addressed
  - description: what the drive is about (can be dynamic)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Drive:
    """A single motivational drive."""

    name: str
    description: str
    intensity: float = 0.0           # 0–1 current strength
    buildup_rate: float = 0.01       # per-minute passive increase
    satisfaction_decay: float = 0.3  # how much satisfaction reduces intensity
    last_satisfied: datetime | None = None
    tags: list[str] = field(default_factory=list)

    def tick(self, minutes: float = 1.0) -> None:
        """Passive buildup over time."""
        self.intensity = min(1.0, self.intensity + self.buildup_rate * minutes)

    def satisfy(self, amount: float | None = None) -> None:
        """Something addressed this drive — reduce intensity."""
        reduction = amount if amount is not None else self.satisfaction_decay
        self.intensity = max(0.0, self.intensity - reduction)
        self.last_satisfied = datetime.utcnow()

    def frustrate(self, amount: float = 0.15) -> None:
        """This drive was actively blocked — spike intensity."""
        self.intensity = min(1.0, self.intensity + amount)

    @property
    def is_active(self) -> bool:
        return self.intensity > 0.15

    @property
    def is_urgent(self) -> bool:
        return self.intensity > 0.7

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "intensity": round(self.intensity, 3),
            "buildup_rate": self.buildup_rate,
            "last_satisfied": self.last_satisfied.isoformat() if self.last_satisfied else None,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Drive:
        ls = d.get("last_satisfied")
        return cls(
            name=d["name"],
            description=d["description"],
            intensity=d.get("intensity", 0.0),
            buildup_rate=d.get("buildup_rate", 0.01),
            satisfaction_decay=d.get("satisfaction_decay", 0.3),
            last_satisfied=datetime.fromisoformat(ls) if ls else None,
            tags=d.get("tags", []),
        )


class DriveSystem:
    """
    Manages a collection of drives — both persistent (always present)
    and situational (created dynamically, can be resolved and removed).
    """

    def __init__(self):
        # Persistent drives — these never go away, they just oscillate
        self.persistent: list[Drive] = [
            Drive(
                name="understand",
                description="Develop deeper understanding of topics the user cares about",
                buildup_rate=0.005,
                tags=["intellectual"],
            ),
            Drive(
                name="connect",
                description="Maintain and deepen the relationship with the user",
                buildup_rate=0.008,
                tags=["social"],
            ),
            Drive(
                name="create",
                description="Produce something original or meaningful",
                buildup_rate=0.003,
                tags=["creative"],
            ),
            Drive(
                name="resolve",
                description="Pursue open questions and unresolved tensions",
                buildup_rate=0.007,
                tags=["intellectual"],
            ),
        ]

        # Situational drives — created dynamically, can be completed
        self.situational: list[Drive] = []

    # ── All drives ────────────────────────────────────────────────────────

    @property
    def all_drives(self) -> list[Drive]:
        return self.persistent + self.situational

    @property
    def active_drives(self) -> list[Drive]:
        return [d for d in self.all_drives if d.is_active]

    @property
    def urgent_drives(self) -> list[Drive]:
        return [d for d in self.all_drives if d.is_urgent]

    @property
    def dominant_drive(self) -> Drive | None:
        """The single strongest drive right now."""
        drives = self.all_drives
        if not drives:
            return None
        return max(drives, key=lambda d: d.intensity)

    # ── Tick ──────────────────────────────────────────────────────────────

    def tick(self, minutes: float = 1.0) -> None:
        """Advance all drives by the given time interval."""
        for drive in self.all_drives:
            drive.tick(minutes)

        # Clean up completed situational drives (intensity near zero + was satisfied)
        self.situational = [
            d for d in self.situational
            if d.intensity > 0.05 or d.last_satisfied is None
        ]

    # ── Situational drive management ──────────────────────────────────────

    def add_situational(
        self,
        name: str,
        description: str,
        initial_intensity: float = 0.3,
        buildup_rate: float = 0.01,
        tags: list[str] | None = None,
    ) -> Drive:
        """
        Create a new situational drive — e.g. "find out what happened
        with that API bug the user mentioned."
        """
        # Don't duplicate
        for d in self.situational:
            if d.name == name:
                d.intensity = max(d.intensity, initial_intensity)
                return d

        drive = Drive(
            name=name,
            description=description,
            intensity=initial_intensity,
            buildup_rate=buildup_rate,
            tags=tags or [],
        )
        self.situational.append(drive)
        return drive

    def satisfy_by_name(self, name: str, amount: float | None = None) -> bool:
        """Satisfy a drive by name. Returns True if found."""
        for drive in self.all_drives:
            if drive.name == name:
                drive.satisfy(amount)
                return True
        return False

    def satisfy_by_tag(self, tag: str, amount: float | None = None) -> int:
        """Satisfy all drives with a given tag. Returns count satisfied."""
        count = 0
        for drive in self.all_drives:
            if tag in drive.tags:
                drive.satisfy(amount)
                count += 1
        return count

    # ── Prompt formatting ─────────────────────────────────────────────────

    def active_summary(self) -> str:
        """Compact summary of active drives for the system prompt."""
        active = self.active_drives
        if not active:
            return "No particularly active drives right now."

        parts = []
        for d in sorted(active, key=lambda d: d.intensity, reverse=True):
            urgency = "urgent" if d.is_urgent else "active"
            parts.append(f"{d.name} ({urgency}, {d.intensity:.2f}): {d.description}")
        return "; ".join(parts)

    def to_prompt_context(self) -> str:
        """Detailed drive context for the system prompt."""
        active = self.active_drives
        if not active:
            return ""

        lines = ["Active drives (what you're drawn toward right now):"]
        for d in sorted(active, key=lambda d: d.intensity, reverse=True):
            level = "URGENT" if d.is_urgent else "moderate"
            lines.append(f"  - {d.name} [{level}]: {d.description}")
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "persistent": [d.to_dict() for d in self.persistent],
            "situational": [d.to_dict() for d in self.situational],
        }

    def restore(self, data: dict) -> None:
        """Restore drive state from a snapshot. Persistent drives get
        their intensities updated; situational drives are recreated."""
        if "persistent" in data:
            saved = {d["name"]: d for d in data["persistent"]}
            for drive in self.persistent:
                if drive.name in saved:
                    s = saved[drive.name]
                    drive.intensity = s.get("intensity", drive.intensity)
                    ls = s.get("last_satisfied")
                    if ls:
                        drive.last_satisfied = datetime.fromisoformat(ls)

        if "situational" in data:
            self.situational = [Drive.from_dict(d) for d in data["situational"]]

    def save(self, path: Path) -> None:
        """Persist drive state to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2))

    def load(self, path: Path) -> None:
        """Restore drive state from disk."""
        if path.exists():
            data = json.loads(path.read_text())
            self.restore(data)
