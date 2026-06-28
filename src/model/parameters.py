"""
Calibrated parameters for the rent-freeze housing market simulation.

Values are chosen to approximate a dense urban rental market (e.g. US metro
renter households, ACS-style income dispersion, landlord operating margins).
Units are normalized so baseline median contract rent equals 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PolicyScenario(str, Enum):
    """Counterfactual policy regimes compared in Monte Carlo experiments."""

    BASELINE = "baseline"
    RENT_FREEZE = "rent_freeze"
    PARTIAL_CAP = "partial_cap"  # cap on renewals only ( vacancy decontrol )


@dataclass(frozen=True)
class ModelParameters:
    """
    Structural parameters for the search-and-matching housing model.

    The model follows a discrete-time overlapping-generations of matches:
    tenants have log preferences over consumption and housing quality;
    landlords choose maintenance to maximize flow profit subject to rent caps.
    """

    # --- Market scale ---
    n_units: int = 2_000
    n_periods: int = 120  # 10 years monthly
    random_seed: int = 42

    # --- Tenant heterogeneity (calibrated to urban renter income dispersion) ---
    # log income ~ N(mu_y, sigma_y); scaled so median post-tax budget ≈ 2.5× median rent
    income_log_mean: float = 0.92
    income_log_std: float = 0.55
    taste_quality_mean: float = 0.35  # θ_i ~ N(·); heterogeneity in quality valuation
    taste_quality_std: float = 0.12
    search_cost: float = 0.015  # per-period utility cost while unhoused/searching
    outside_option_utility: float = -2.5  # staying with family / leaving metro
    mobility_rate: float = 0.025  # exogenous separation hazard per occupied match / period
    arrival_rate: float = 0.018  # new entrants per period as fraction of market size

    # --- Housing quality & maintenance (Glaeser-Luttmer style reduced form) ---
    initial_quality_mean: float = 1.0
    initial_quality_std: float = 0.15
    quality_depreciation: float = 0.10  # ρ: natural decay if m = 0
    maintenance_cost_coef: float = 9.0  # κ in c(m) = κ m²/2
    quality_maintenance_elasticity: float = 0.50  # q' = m^elasticity (concave upgrade)
    min_quality: float = 0.25
    max_quality: float = 1.45

    # --- Landlord technology & outside option ---
    operating_fixed_cost: float = 0.18  # taxes, insurance, per period
    vacancy_cost: float = 0.04  # holding cost while vacant
    conversion_prob_base: float = 0.001  # hazard of removing unit from rental stock
    # conversion rises when net operating income is persistently negative

    # --- Pricing (baseline competitive-ish clearing) ---
    baseline_rent_markup: float = 0.18  # over marginal cost at t=0
    rent_adjustment_speed: float = 0.12  # partial adjustment toward WTP in baseline

    # --- Search & matching (random search with directed WTP ranking) ---
    applications_per_searcher: int = 3  # limited consideration set (realistic frictions)
    match_tie_noise: float = 0.05  # logistic taste shocks in application ranking
    max_vacancy_fill_rate: float = 0.55  # share of vacant units matched per period (friction)

    # --- Rent freeze policy ---
    freeze_start_period: int = 24  # policy enacted after burn-in
    freeze_cap_level: float = 0.88  # binding cap in rising-rent environment
    allow_vacancy_decontrol: bool = False  # if True, new leases above cap allowed
    grandfather_incumbent: bool = True  # incumbents keep pre-freeze rent if lower
    maintenance_penalty_threshold: float = 0.15  # NOI margin triggering quality collapse

    max_rent_to_income: float = 0.45  # affordability constraint (HUD-style stress threshold)
    n_replications: int = 20
    burn_in_periods: int = 24

    def validate(self) -> None:
        if self.n_units <= 0:
            raise ValueError("n_units must be positive")
        if not 0 < self.mobility_rate < 1:
            raise ValueError("mobility_rate must lie in (0, 1)")
        if self.freeze_cap_level <= 0:
            raise ValueError("freeze_cap_level must be positive")


@dataclass
class SimulationConfig:
    """Runtime configuration bundling parameters with scenario label."""

    parameters: ModelParameters = field(default_factory=ModelParameters)
    scenario: PolicyScenario = PolicyScenario.BASELINE
    label: Optional[str] = None

    def __post_init__(self) -> None:
        self.parameters.validate()
        if self.label is None:
            self.label = self.scenario.value
