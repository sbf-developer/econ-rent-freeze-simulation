"""Sanity checks for model invariants and policy logic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.agents import Landlord, Tenant
from src.model.market import HousingMarket
from src.model.parameters import ModelParameters, PolicyScenario
from src.model.policy import apply_policy


def test_affordability_constraint() -> None:
    params = ModelParameters(n_units=50, n_periods=5, n_replications=1)
    tenant = Tenant(tenant_id=0, income=3.0, taste_quality=0.3)
    u_low = tenant.flow_utility(0.8, 1.0, params)
    u_high = tenant.flow_utility(1.4, 1.0, params)
    assert np.isfinite(u_low)
    assert u_low > u_high
    assert not np.isfinite(tenant.flow_utility(1.5, 1.0, params))


def test_rent_cap_enforcement() -> None:
    params = ModelParameters(freeze_cap_level=0.88, freeze_start_period=2)
    policy = apply_policy(PolicyScenario.RENT_FREEZE, params)
    assert policy.allowed_rent(1.2, incumbent=True, current_rent=1.2, period=5) == 0.88


def test_maintenance_falls_under_cap() -> None:
    params = ModelParameters()
    landlord = Landlord(landlord_id=0, unit_id=0, maintenance_cost_coef=params.maintenance_cost_coef)
    shadow = Landlord.hedonic_market_rent(1.2, params)
    m_uncapped = landlord.optimal_maintenance(1.4, 1.2, False, params, shadow_rent=shadow)
    m_capped = landlord.optimal_maintenance(
        1.4, 1.2, False, params, rent_cap=0.88, shadow_rent=shadow
    )
    assert m_capped <= m_uncapped


def test_simulation_runs() -> None:
    params = ModelParameters(n_units=100, n_periods=10, n_replications=1)
    policy = apply_policy(PolicyScenario.BASELINE, params)
    market = HousingMarket(params, policy, np.random.default_rng(0))
    history = market.run(10)
    assert len(history) == 10
    assert 0 <= history[-1].vacancy_rate <= 1


if __name__ == "__main__":
    test_affordability_constraint()
    test_rent_cap_enforcement()
    test_maintenance_falls_under_cap()
    test_simulation_runs()
    print("All sanity checks passed.")
