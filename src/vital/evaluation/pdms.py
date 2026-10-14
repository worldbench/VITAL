"""Standalone NAVSIM v1 PDM-score aggregation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ALIASES = {
    "nc": ("no_at_fault_collisions", "NC", "nc"),
    "dac": ("drivable_area_compliance", "DAC", "dac"),
    "ddc": ("driving_direction_compliance", "DDC", "ddc"),
    "tlc": ("traffic_light_compliance", "TLC", "tlc"),
    "ep": ("ego_progress", "EP", "ep"),
    "ttc": ("time_to_collision_within_bound", "TTC", "ttc"),
    "lk": ("lane_keeping", "LK", "lk"),
    "comfort": ("history_comfort", "comfort", "C", "c"),
    "extended_comfort": ("two_frame_extended_comfort", "extended_comfort"),
}


def _value(row: Mapping[str, Any], key: str, *, default: float | None = None) -> float:
    for name in ALIASES[key]:
        if name in row and row[name] not in (None, "", "nan", "NaN"):
            value = float(row[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")
            return value
    if default is not None:
        return default
    raise KeyError(f"Missing score component {key}; accepted names={ALIASES[key]}")


def pdms_components(row: Mapping[str, Any]) -> dict[str, float]:
    """Compute compact v1 PDMS and expose components on a 0--100 scale."""

    nc, dac = _value(row, "nc"), _value(row, "dac")
    ddc, tlc = _value(row, "ddc", default=1.0), _value(row, "tlc", default=1.0)
    ep, ttc, comfort = _value(row, "ep"), _value(row, "ttc"), _value(row, "comfort")
    lk = _value(row, "lk", default=1.0)
    weighted = [(ep, 5.0), (ttc, 5.0), (comfort, 2.0)]
    score = 100.0 * nc * dac * sum(value * weight for value, weight in weighted) / sum(weight for _, weight in weighted)
    return {"NC": 100.0 * nc, "DAC": 100.0 * dac, "DDC": 100.0 * ddc, "TLC": 100.0 * tlc,
            "EP": 100.0 * ep, "TTC": 100.0 * ttc, "LK": 100.0 * lk, "C": 100.0 * comfort,
            "PDMS": score}
