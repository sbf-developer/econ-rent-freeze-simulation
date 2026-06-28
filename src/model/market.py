"""
Period-by-period housing market clearing with search frictions.

Each period:
  1. Exogenous separations and new tenant arrivals
  2. Landlords choose maintenance; quality updates
  3. Vacant units posted; searching tenants apply (limited consideration)
  4. Landlords accept highest effective bid subject to policy cap
  5. Baseline regime: incumbents' rents partially adjust toward WTP
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .agents import (
    HousingUnit,
    Landlord,
    Tenant,
    draw_tenant_characteristics,
    initialize_units,
    update_quality,
)
from .parameters import ModelParameters
from .policy import RentFreezePolicy


@dataclass
class PeriodMetrics:
    """Aggregates recorded each simulation period."""

    period: int
    mean_rent: float
    median_rent: float
    mean_quality: float
    vacancy_rate: float
    searchers: int
    mean_tenant_utility: float
    mean_landlord_profit: float
    misallocation_index: float  # E[(θ_i - θ̄)(q_i - q̄)] mismatch
    conversion_count: int
    active_units: int


@dataclass
class MarketState:
    """Full micro state of the housing market."""

    units: list[HousingUnit]
    landlords: list[Landlord]
    tenants: dict[int, Tenant]
    next_tenant_id: int = 0
    policy: RentFreezePolicy = field(default_factory=lambda: RentFreezePolicy(False, 1.0, 0))

    def active_units(self) -> list[HousingUnit]:
        return [u for u in self.units if u.active]

    def searching_tenants(self) -> list[Tenant]:
        return [
            t
            for t in self.tenants.values()
            if t.active and t.matched_unit_id is None
        ]


class HousingMarket:
    """Agent-based housing market simulator."""

    def __init__(
        self,
        params: ModelParameters,
        policy: RentFreezePolicy,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.params = params
        self.policy = policy
        self.rng = rng or np.random.default_rng(params.random_seed)

        units, landlords = initialize_units(params.n_units, params, self.rng)
        self.state = MarketState(units=units, landlords=landlords, tenants={}, policy=policy)

        self._initialize_population()

    def _initialize_population(self) -> None:
        """Match initial tenants to units via assortative matching on income/rent."""
        params = self.params
        state = self.state
        active = sorted(state.active_units(), key=lambda u: u.contract_rent, reverse=True)
        n_match = int(0.94 * len(active))

        chars = draw_tenant_characteristics(n_match, params, self.rng)
        # Highest income households match to highest rent units (stable sort)
        chars.sort(key=lambda x: x[0], reverse=True)

        for unit, (inc, taste) in zip(active[:n_match], chars):
            max_affordable = params.max_rent_to_income * inc
            if unit.contract_rent > max_affordable:
                unit.contract_rent = float(max(0.5, max_affordable * 0.92))

            tid = state.next_tenant_id
            state.next_tenant_id += 1
            tenant = Tenant(
                tenant_id=tid,
                income=inc,
                taste_quality=taste,
                matched_unit_id=unit.unit_id,
            )
            state.tenants[tid] = tenant
            unit.tenant_id = tid
            unit.incumbent = True

    def _add_entrants(self) -> None:
        params = self.params
        n_new = self.rng.poisson(params.arrival_rate * params.n_units)
        if n_new == 0:
            return

        for inc, taste in draw_tenant_characteristics(n_new, params, self.rng):
            tid = self.state.next_tenant_id
            self.state.next_tenant_id += 1
            self.state.tenants[tid] = Tenant(
                tenant_id=tid,
                income=inc,
                taste_quality=taste,
            )

    def _process_separations(self) -> list[HousingUnit]:
        """Exogenous match dissolution; returns newly vacant units."""
        params = self.params
        vacated: list[HousingUnit] = []

        for unit in self.state.active_units():
            if not unit.is_occupied:
                continue
            if self.rng.random() < params.mobility_rate:
                tid = unit.tenant_id
                assert tid is not None
                tenant = self.state.tenants[tid]
                tenant.matched_unit_id = None
                tenant.periods_searching = 0
                unit.tenant_id = None
                unit.periods_vacant = 0
                unit.incumbent = False
                vacated.append(unit)

        return vacated

    def _landlord_maintenance_phase(self, period: int) -> None:
        params = self.params
        cap = self.policy.cap_level if self.policy.active and period >= self.policy.start_period else None

        for landlord in self.state.landlords:
            unit = self.state.units[landlord.unit_id]
            if not unit.active:
                continue

            allowed = self.policy.allowed_rent(
                unit.contract_rent,
                unit.incumbent,
                unit.contract_rent,
                period,
            )
            shadow = Landlord.hedonic_market_rent(unit.quality, params)
            m = landlord.optimal_maintenance(
                rent=unit.contract_rent,
                quality=unit.quality,
                vacant=unit.is_vacant,
                params=params,
                rent_cap=allowed if cap is not None else None,
                shadow_rent=shadow,
            )
            unit.quality = update_quality(unit.quality, m, params)
            unit.last_maintenance = m
            profit = landlord.flow_profit(unit.contract_rent if not unit.is_vacant else 0.0, m, unit.is_vacant, params)
            unit.cumulative_noi += profit

            # Conversion hazard increases when cumulative NOI deeply negative
            if unit.cumulative_noi < -0.5:
                hazard = params.conversion_prob_base * (1 + abs(unit.cumulative_noi))
                if self.rng.random() < hazard:
                    unit.active = False
                    if unit.tenant_id is not None:
                        t = self.state.tenants[unit.tenant_id]
                        t.matched_unit_id = None
                        unit.tenant_id = None

    def _baseline_rent_adjustment(self, period: int) -> None:
        """Partial rent adjustment for occupied units (baseline market only)."""
        if self.policy.active and period >= self.policy.start_period:
            return

        params = self.params
        for unit in self.state.active_units():
            if not unit.is_occupied:
                continue
            tid = unit.tenant_id
            assert tid is not None
            tenant = self.state.tenants[tid]
            wtp = tenant.willingness_to_pay(unit.quality, params, reference_rent=unit.contract_rent)
            max_affordable = params.max_rent_to_income * tenant.income
            target = 0.7 * unit.contract_rent + 0.3 * min(wtp, max_affordable)
            unit.contract_rent += params.rent_adjustment_speed * (target - unit.contract_rent)
            unit.contract_rent = float(
                np.clip(unit.contract_rent, 0.4, min(3.0, max_affordable))
            )

    def _purge_inactive_tenants(self) -> None:
        """Drop exited searchers to keep matching loops bounded."""
        inactive = [tid for tid, t in self.state.tenants.items() if not t.active]
        for tid in inactive:
            del self.state.tenants[tid]

    def _match_searchers(self, period: int) -> None:
        params = self.params
        vacant = [u for u in self.state.active_units() if u.is_vacant]

        for unit in vacant:
            unit.periods_vacant += 1

        searchers = self.state.searching_tenants()
        if not searchers or not vacant:
            return

        # Process only as many searchers as the vacancy flow can plausibly absorb.
        queue_limit = max(len(vacant) * 4, 20)
        if len(searchers) > queue_limit:
            self.rng.shuffle(searchers)
            searchers = searchers[:queue_limit]

        self.rng.shuffle(searchers)

        # Realistic friction: not all vacant units fill immediately each month
        max_fills = max(1, int(len(vacant) * params.max_vacancy_fill_rate))
        fills_remaining = max_fills

        for tenant in searchers:
            if not vacant or fills_remaining <= 0:
                break

            # Limited consideration set — realistic search friction
            n_apps = min(params.applications_per_searcher, len(vacant))
            candidates = self.rng.choice(vacant, size=n_apps, replace=False)

            best_unit: Optional[HousingUnit] = None
            best_score = -np.inf

            for unit in candidates:
                proposed_rent = tenant.willingness_to_pay(unit.quality, params) * 0.92
                max_affordable = params.max_rent_to_income * tenant.income
                proposed_rent = min(proposed_rent, max_affordable)
                allowed = self.policy.allowed_rent(
                    proposed_rent,
                    incumbent=False,
                    current_rent=unit.contract_rent,
                    period=period,
                )
                u_flow = tenant.flow_utility(allowed, unit.quality, params)
                if not np.isfinite(u_flow) or u_flow <= params.outside_option_utility:
                    continue

                noise = self.rng.gumbel(0, params.match_tie_noise)
                score = u_flow + noise

                min_rent = Landlord.reservation_rent(unit.quality, params)
                if allowed >= min_rent and score > best_score:
                    best_score = score
                    best_unit = unit

            if best_unit is None:
                tenant.periods_searching += 1
                # Long search: exit market (move away / double up)
                if tenant.periods_searching > 12:
                    tenant.active = False
                continue

            proposed = tenant.willingness_to_pay(best_unit.quality, params) * 0.92
            proposed = min(proposed, params.max_rent_to_income * tenant.income)
            rent = self.policy.allowed_rent(proposed, False, best_unit.contract_rent, period)
            best_unit.contract_rent = rent
            best_unit.tenant_id = tenant.tenant_id
            best_unit.incumbent = period >= self.policy.start_period and self.policy.active
            tenant.matched_unit_id = best_unit.unit_id
            tenant.periods_searching = 0
            vacant.remove(best_unit)
            fills_remaining -= 1

    def _compute_misallocation(self) -> float:
        """
        Covariance-based mismatch: high-θ tenants in low-q units (and vice versa)
        relative to cross-sectional means. Positive values imply inefficient sorting.
        """
        thetas = []
        qualities = []
        for unit in self.state.active_units():
            if not unit.is_occupied:
                continue
            t = self.state.tenants[unit.tenant_id]  # type: ignore[index]
            thetas.append(t.taste_quality)
            qualities.append(unit.quality)

        if len(thetas) < 2:
            return 0.0

        theta_arr = np.array(thetas)
        q_arr = np.array(qualities)
        return float(np.mean((theta_arr - theta_arr.mean()) * (q_arr - q_arr.mean())))

    def _enforce_rent_caps(self, period: int) -> None:
        """Apply statutory caps to all active lease contracts when policy is in force."""
        if not self.policy.active or period < self.policy.start_period:
            return

        for unit in self.state.active_units():
            unit.contract_rent = self.policy.allowed_rent(
                unit.contract_rent,
                incumbent=unit.incumbent,
                current_rent=unit.contract_rent,
                period=period,
            )

    def step(self, period: int) -> PeriodMetrics:
        """Advance one period; return aggregate metrics."""
        self._add_entrants()
        self._process_separations()
        self._enforce_rent_caps(period)
        self._landlord_maintenance_phase(period)
        self._baseline_rent_adjustment(period)
        self._match_searchers(period)
        if period % 12 == 0:
            self._purge_inactive_tenants()

        return self.collect_metrics(period)

    def collect_metrics(self, period: int) -> PeriodMetrics:
        active = self.state.active_units()
        occupied = [u for u in active if u.is_occupied]

        rents = [u.contract_rent for u in occupied] if occupied else [0.0]
        qualities = [u.quality for u in active] if active else [1.0]

        tenant_utils = []
        landlord_profits = []

        for unit in occupied:
            t = self.state.tenants[unit.tenant_id]  # type: ignore[index]
            u = t.flow_utility(unit.contract_rent, unit.quality, self.params)
            if np.isfinite(u):
                tenant_utils.append(u)

        for landlord in self.state.landlords:
            unit = self.state.units[landlord.unit_id]
            if not unit.active:
                continue
            m = unit.last_maintenance
            landlord_profits.append(
                landlord.flow_profit(
                    unit.contract_rent if unit.is_occupied else 0.0,
                    m,
                    unit.is_vacant,
                    self.params,
                )
            )

        conversions = sum(1 for u in self.state.units if not u.active)

        return PeriodMetrics(
            period=period,
            mean_rent=float(np.mean(rents)),
            median_rent=float(np.median(rents)),
            mean_quality=float(np.mean(qualities)),
            vacancy_rate=1 - len(occupied) / max(len(active), 1),
            searchers=len(self.state.searching_tenants()),
            mean_tenant_utility=float(np.mean(tenant_utils)) if tenant_utils else 0.0,
            mean_landlord_profit=float(np.mean(landlord_profits)) if landlord_profits else 0.0,
            misallocation_index=self._compute_misallocation(),
            conversion_count=conversions,
            active_units=len(active),
        )

    def run(self, n_periods: Optional[int] = None) -> list[PeriodMetrics]:
        n = n_periods or self.params.n_periods
        history: list[PeriodMetrics] = []
        for t in range(n):
            history.append(self.step(t))
        return history
