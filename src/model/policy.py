"""Rent control policy implementation and enforcement rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .parameters import ModelParameters, PolicyScenario


@dataclass(frozen=True)
class RentFreezePolicy:
    """
    Hard rent freeze: contract rent cannot exceed cap for incumbent leases.

    Optional vacancy decontrol permits market rents on turnover.
    """

    active: bool
    cap_level: float
    start_period: int
    grandfather_incumbent: bool = True
    vacancy_decontrol: bool = False

    def allowed_rent(
        self,
        proposed_rent: float,
        incumbent: bool,
        current_rent: float,
        period: int,
    ) -> float:
        if not self.active or period < self.start_period:
            return proposed_rent

        if self.vacancy_decontrol and not incumbent:
            return proposed_rent

        if self.grandfather_incumbent and incumbent:
            return min(current_rent, self.cap_level)

        return min(proposed_rent, self.cap_level)


def apply_policy(
    scenario: PolicyScenario,
    params: ModelParameters,
) -> RentFreezePolicy:
    """Construct policy object from scenario enum and parameters."""
    if scenario == PolicyScenario.BASELINE:
        return RentFreezePolicy(
            active=False,
            cap_level=1e9,
            start_period=params.freeze_start_period,
        )

    if scenario == PolicyScenario.RENT_FREEZE:
        return RentFreezePolicy(
            active=True,
            cap_level=params.freeze_cap_level,
            start_period=params.freeze_start_period,
            grandfather_incumbent=params.grandfather_incumbent,
            vacancy_decontrol=False,
        )

    if scenario == PolicyScenario.PARTIAL_CAP:
        return RentFreezePolicy(
            active=True,
            cap_level=params.freeze_cap_level,
            start_period=params.freeze_start_period,
            grandfather_incumbent=True,
            vacancy_decontrol=True,
        )

    raise ValueError(f"Unknown scenario: {scenario}")
