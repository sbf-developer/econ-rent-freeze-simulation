#!/usr/bin/env python3
"""
Run the rent-freeze counterfactual experiment suite and generate figures.

Usage:
    python scripts/run_simulation.py [--fast]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.parameters import ModelParameters, PolicyScenario
from src.paper_export import export_paper_stats
from src.simulation import SimulationConfig, run_experiment, save_results
from src.visualization import generate_all_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rent freeze ABM simulation")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Reduced scale for quick smoke test (500 units, 10 replications)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Paper-scale run (2,000 units, 20 replications, 120 periods)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output",
        help="Output directory for CSV and figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = ModelParameters()

    if args.fast:
        params = ModelParameters(
            n_units=500,
            n_periods=60,
            n_replications=10,
            freeze_start_period=12,
            burn_in_periods=12,
        )
    elif args.paper:
        params = ModelParameters(
            n_units=2_000,
            n_periods=120,
            n_replications=20,
            freeze_start_period=24,
            burn_in_periods=24,
        )

    print("Running counterfactual suite...")
    print(f"  Units: {params.n_units:,} | Periods: {params.n_periods} | Replications: {params.n_replications}")
    print("  (This may take several minutes at full scale.)")

    scenarios = [
        PolicyScenario.BASELINE,
        PolicyScenario.RENT_FREEZE,
        PolicyScenario.PARTIAL_CAP,
    ]
    results = {}
    for i, scenario in enumerate(scenarios, 1):
        print(f"  [{i}/{len(scenarios)}] {scenario.value}...", flush=True)
        config = SimulationConfig(parameters=params, scenario=scenario)
        results[scenario.value] = run_experiment(config, parallel=not args.fast)
    save_results(results, args.output)

    figure_paths = generate_all_figures(
        results,
        args.output / "figures",
        freeze_start=params.freeze_start_period,
        horizon=params.n_periods - 1,
    )

    if args.paper:
        export_paper_stats(
            results,
            ROOT / "paper",
            params.n_periods - 1,
            params.freeze_start_period,
        )
        print("  Exported paper/generated_*.tex")

    print("\nResults saved to:", args.output)
    print("Figures:")
    for p in figure_paths:
        print(f"  {p}")

    # Terminal summary table
    terminal = params.n_periods - 1
    print(f"\n{'Scenario':<25} {'Rent':>8} {'Quality':>8} {'Vacancy':>8} {'Utility':>8}")
    print("-" * 60)
    for key, res in results.items():
        row = res.summary.loc[res.summary["period"] == terminal].iloc[0]
        print(
            f"{key:<25} {row['mean_rent']:8.3f} {row['mean_quality']:8.3f} "
            f"{row['vacancy_rate']:8.3f} {row['mean_tenant_utility']:8.3f}"
        )


if __name__ == "__main__":
    main()
