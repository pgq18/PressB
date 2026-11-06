"""Pairwise centre-distance limits for randomized tabletop button layouts."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math


SPACING_PROFILES = {
    "near": {"min_separation": .06, "max_separation": .10},
    "medium": {"min_separation": .10, "max_separation": .18},
    "far": {"min_separation": .18, "max_separation": .30},
}


@dataclass(frozen=True)
class ButtonSpacing:
    profile: str | None
    min_separation: float
    max_separation: float | None

    def accepts(self, distance: float) -> bool:
        return self.min_separation <= distance and (
            self.max_separation is None or distance <= self.max_separation)


def resolve_spacing(sampling: Mapping, button_radius: float = .022) -> ButtonSpacing:
    """Resolve a named band or the legacy minimum-only spacing configuration.

    A selected profile replaces the top-level separation limits. Its two bounds
    may be customized together in ``spacing_profiles``. Distances are in metres
    between centres, and apply to every pair, not just nearest neighbours.
    """
    profile = sampling.get("spacing_profile")
    if profile is not None and (not isinstance(profile, str) or profile not in SPACING_PROFILES):
        raise ValueError("spacing_profile must be near, medium, far, or null")
    limits = sampling
    if profile is not None:
        overrides = sampling.get("spacing_profiles", {})
        if not isinstance(overrides, Mapping) or set(overrides) - SPACING_PROFILES.keys():
            raise ValueError("spacing_profiles must map near, medium, or far to distance limits")
        limits = overrides.get(profile, SPACING_PROFILES[profile])
        if not isinstance(limits, Mapping) or not {"min_separation", "max_separation"} <= limits.keys():
            raise ValueError("A spacing profile requires min_separation and max_separation")
        if limits["max_separation"] is None:
            raise ValueError("A spacing profile requires a finite max_separation")
    minimum = float(limits.get("min_separation", .10))
    maximum = limits.get("max_separation")
    maximum = None if maximum is None else float(maximum)
    if not math.isfinite(button_radius) or button_radius <= 0:
        raise ValueError("Button radius must be positive and finite")
    if not math.isfinite(minimum) or minimum < 2 * button_radius:
        raise ValueError("Minimum separation must leave button housings disjoint")
    if maximum is not None and (not math.isfinite(maximum) or maximum <= minimum):
        raise ValueError("max_separation must be finite and greater than min_separation")
    return ButtonSpacing(profile, minimum, maximum)
