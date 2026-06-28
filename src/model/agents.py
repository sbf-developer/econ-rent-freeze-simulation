"""
Heterogeneous landlord and tenant agents with utility-maximizing behavior.

Tenants: quasi-linear log utility with idiosyncratic quality taste.
Landlords: profit-maximizing maintenance choice given rent constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .parameters import ModelParameters


@dataclass
class Tenant:
    """Renter household searching for housing or matched to a unit."""

    tenant_id: int
    income: float  # monthly budget net of non-housing essentials
    taste_quality: float  # θ_i scaling log(q) in utility
    matched_unit_id: Optional[int] = None
    periods_searching: int = 0
    active: bool = True

    def flow_utility(
        self,
        rent: float,
        quality: float,
        params: ModelParameters,
        searching: bool = False,
    ) -> float:
        """
        Per-period utility U = log(y - r) + θ log(q) - search cost if searching.

        Returns -inf if rent exceeds income (infeasible).
        """
        disposable = self.income - rent
        if disposable <= (1.0 - params.max_rent_to_income) * self.income:
            return -np.inf

        q = np.clip(quality, params.min_quality, params.max_quality)
        utility = np.log(disposable) + self.taste_quality * np.log(q)

        if searching:
            utility -= params.search_cost * (1 + 0.1 * self.periods_searching)

        return float(utility)

    def willingness_to_pay(
        self,
        quality: float,
        params: ModelParameters,
        reference_rent: float = 1.0,
    ) -> float:
        """
        Inverse demand: rent r such that U(r, q) equals utility at reference rent
        on a unit of reference quality (1.0), holding outside option fixed.
        """
        u_ref = self.flow_utility(reference_rent, 1.0, params)
        q = np.clip(quality, params.min_quality, params.max_quality)

        # log(y - r) = u_ref - θ log(q)  =>  r = y - exp(u_ref - θ log(q))
        target = u_ref - self.taste_quality * np.log(q)
        wtp = self.income - np.exp(target)
        return float(np.clip(wtp, 0.05, 0.95 * self.income))


@dataclass
class HousingUnit:
    """Rental dwelling with endogenous quality and contract rent."""

    unit_id: int
    quality: float
    contract_rent: float
    tenant_id: Optional[int] = None
    landlord_id: int = 0
    incumbent: bool = False  # matched before policy; affects grandfathering
    periods_vacant: int = 0
    active: bool = True  # False if converted out of rental stock
    cumulative_noi: float = 0.0  # for conversion decisions
    last_maintenance: float = 0.0  # most recent upkeep choice (for metrics)

    @property
    def is_vacant(self) -> bool:
        return self.tenant_id is None and self.active

    @property
    def is_occupied(self) -> bool:
        return self.tenant_id is not None and self.active


@dataclass
class Landlord:
    """
    Small landlord (1 unit) choosing maintenance each period.

    Reduced-form profit: π = r - κ m²/2 - δ - vacancy cost if empty.
    """

    landlord_id: int
    unit_id: int
    maintenance_cost_coef: float = 1.8
    operating_fixed_cost: float = 0.08

    def optimal_maintenance(
        self,
        rent: float,
        quality: float,
        vacant: bool,
        params: ModelParameters,
        rent_cap: Optional[float] = None,
        shadow_rent: Optional[float] = None,
    ) -> float:
        """
        Choose m ≥ 0 to maximize r - c(m) accounting for quality transition.

        Under binding cap, landlords cut maintenance when marginal return falls
        below marginal cost — the primary channel for quality deterioration.
        """
        effective_rent = rent
        if rent_cap is not None:
            effective_rent = min(rent, rent_cap)

        net_margin = effective_rent - params.operating_fixed_cost
        if vacant:
            net_margin -= params.vacancy_cost

        if net_margin <= 0:
            return 0.0

        # FOC from quality transition q' = (1-ρ)q + m^γ
        # marginal benefit ≈ (effective_rent + future value) * γ m^{γ-1}
        # simplified myopic: m* = ((net_margin * γ) / κ)^{1/(2-γ)} capped
        gamma = params.quality_maintenance_elasticity
        kappa = self.maintenance_cost_coef

        if net_margin < params.maintenance_penalty_threshold:
            # distressed landlord — sharply reduced upkeep
            scale = max(net_margin / params.maintenance_penalty_threshold, 0.0)
            m_star = scale * 0.15
        else:
            exponent = 1.0 / (2.0 - gamma)
            m_star = ((net_margin * gamma) / kappa) ** exponent

        # When the cap binds below the hedonic shadow rent, upkeep is deferred.
        benchmark = shadow_rent if shadow_rent is not None else rent
        if rent_cap is not None and benchmark > effective_rent + 1e-6:
            squeeze = (effective_rent / benchmark) ** 1.25
            m_star *= float(np.clip(squeeze, 0.05, 1.0))

        return float(np.clip(m_star, 0.0, 0.05))

    @staticmethod
    def hedonic_market_rent(quality: float, params: ModelParameters) -> float:
        """Quality-adjusted market rent index (hedonic benchmark for shadow pricing)."""
        q = np.clip(quality, params.min_quality, params.max_quality)
        return float(0.32 + 0.38 * q + params.operating_fixed_cost)

    @staticmethod
    def reservation_rent(quality: float, params: ModelParameters) -> float:
        """Minimum rent a landlord accepts before keeping the unit vacant."""
        q = np.clip(quality, params.min_quality, params.max_quality)
        return float(params.operating_fixed_cost + 0.06 + 0.10 * q)

    def flow_profit(
        self,
        rent: float,
        maintenance: float,
        vacant: bool,
        params: ModelParameters,
    ) -> float:
        cost = 0.5 * self.maintenance_cost_coef * maintenance**2
        cost += params.operating_fixed_cost
        if vacant:
            cost += params.vacancy_cost
        return float(rent - cost)


def update_quality(
    quality: float,
    maintenance: float,
    params: ModelParameters,
) -> float:
    """Quality law of motion: q_{t+1} = (1-ρ)q_t + m^γ, clipped to bounds."""
    gamma = params.quality_maintenance_elasticity
    q_next = (1 - params.quality_depreciation) * quality + maintenance**gamma
    return float(np.clip(q_next, params.min_quality, params.max_quality))


def draw_tenant_characteristics(
    n: int,
    params: ModelParameters,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Draw (income, taste_quality) pairs for new tenant entrants."""
    incomes = rng.lognormal(params.income_log_mean, params.income_log_std, size=n)
    # Scale incomes so median ≈ 2.5 (median rent = 1.0 in normalized units)
    median_target = 2.5
    incomes *= median_target / np.median(incomes)

    tastes = rng.normal(params.taste_quality_mean, params.taste_quality_std, size=n)
    tastes = np.clip(tastes, 0.05, 0.85)

    return list(zip(incomes.tolist(), tastes.tolist()))


def initialize_units(
    n_units: int,
    params: ModelParameters,
    rng: np.random.Generator,
) -> tuple[list[HousingUnit], list[Landlord]]:
    """Create initial housing stock with dispersed quality and rents."""
    qualities = rng.normal(
        params.initial_quality_mean,
        params.initial_quality_std,
        size=n_units,
    )
    qualities = np.clip(qualities, params.min_quality, params.max_quality)

    # Initial rents: increasing in quality plus idiosyncratic shock
    rents = qualities * (1 + params.baseline_rent_markup)
    rents += rng.normal(0, 0.05, size=n_units)
    rents = np.clip(rents, 0.5, 2.5)

    units: list[HousingUnit] = []
    landlords: list[Landlord] = []

    for i in range(n_units):
        units.append(
            HousingUnit(
                unit_id=i,
                quality=float(qualities[i]),
                contract_rent=float(rents[i]),
                landlord_id=i,
            )
        )
        landlords.append(
            Landlord(
                landlord_id=i,
                unit_id=i,
                maintenance_cost_coef=params.maintenance_cost_coef,
                operating_fixed_cost=params.operating_fixed_cost,
            )
        )

    return units, landlords
