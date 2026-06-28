"""Monte Carlo experiment runner and results aggregation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .model.market import HousingMarket, PeriodMetrics
from .model.parameters import ModelParameters, PolicyScenario, SimulationConfig
from .model.policy import apply_policy


def _run_replication_task(args: tuple) -> list[PeriodMetrics]:
    """Top-level worker for parallel Monte Carlo replications."""
    params, scenario_value, seed = args
    scenario = PolicyScenario(scenario_value)
    return run_single_replication(params, scenario, seed)


@dataclass
class ExperimentResult:
    """Container for one scenario's replicated simulation output."""

    scenario: PolicyScenario
    label: str
    metrics: pd.DataFrame  # long format: replication × period
    summary: pd.DataFrame  # cross-replication means by period


def _metrics_to_records(
    history: list[PeriodMetrics],
    replication: int,
    scenario: PolicyScenario,
) -> list[dict]:
    records = []
    for m in history:
        row = {
            "replication": replication,
            "scenario": scenario.value,
            **m.__dict__,
        }
        records.append(row)
    return records


def run_single_replication(
    params: ModelParameters,
    scenario: PolicyScenario,
    seed: int,
) -> list[PeriodMetrics]:
    policy = apply_policy(scenario, params)
    rng = np.random.default_rng(seed)
    market = HousingMarket(params, policy, rng)
    return market.run(params.n_periods)


def run_experiment(
    config: SimulationConfig,
    n_replications: Optional[int] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> ExperimentResult:
    params = config.parameters
    n_rep = n_replications or params.n_replications
    all_records: list[dict] = []

    tasks = [
        (params, config.scenario.value, params.random_seed + rep * 997)
        for rep in range(n_rep)
    ]

    if parallel and n_rep > 1:
        workers = max_workers or min(n_rep, 4)
        histories: list[list[PeriodMetrics]] = [[] for _ in range(n_rep)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_replication_task, t): i for i, t in enumerate(tasks)}
            for future in as_completed(futures):
                idx = futures[future]
                histories[idx] = future.result()
        for rep, history in enumerate(histories):
            all_records.extend(_metrics_to_records(history, rep, config.scenario))
    else:
        for rep, task in enumerate(tasks):
            history = _run_replication_task(task)
            all_records.extend(_metrics_to_records(history, rep, config.scenario))

    metrics = pd.DataFrame(all_records)
    summary = (
        metrics.groupby("period", as_index=False)
        .agg(
            mean_rent=("mean_rent", "mean"),
            median_rent=("median_rent", "mean"),
            mean_quality=("mean_quality", "mean"),
            vacancy_rate=("vacancy_rate", "mean"),
            searchers=("searchers", "mean"),
            mean_tenant_utility=("mean_tenant_utility", "mean"),
            mean_landlord_profit=("mean_landlord_profit", "mean"),
            misallocation_index=("misallocation_index", "mean"),
            active_units=("active_units", "mean"),
        )
    )
    summary["scenario"] = config.scenario.value

    return ExperimentResult(
        scenario=config.scenario,
        label=config.label or config.scenario.value,
        metrics=metrics,
        summary=summary,
    )


def run_counterfactual_suite(
    params: Optional[ModelParameters] = None,
    scenarios: Optional[list[PolicyScenario]] = None,
) -> dict[str, ExperimentResult]:
    """Run baseline, full freeze, and vacancy-decontrol scenarios."""
    p = params or ModelParameters()
    scenario_list = scenarios or [
        PolicyScenario.BASELINE,
        PolicyScenario.RENT_FREEZE,
        PolicyScenario.PARTIAL_CAP,
    ]

    results: dict[str, ExperimentResult] = {}
    for scenario in scenario_list:
        config = SimulationConfig(parameters=p, scenario=scenario)
        results[scenario.value] = run_experiment(config)
    return results


def save_results(results: dict[str, ExperimentResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        result.summary.to_csv(output_dir / f"summary_{name}.csv", index=False)
        result.metrics.to_csv(output_dir / f"metrics_{name}.csv", index=False)
