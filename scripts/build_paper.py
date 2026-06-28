#!/usr/bin/env python3
"""Run simulation, export LaTeX stats, and compile the paper PDF."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.parameters import ModelParameters, PolicyScenario
from src.paper_export import export_paper_stats
from src.simulation import SimulationConfig, run_experiment, save_results
from src.visualization import generate_all_figures


def main() -> None:
    t0 = time.perf_counter()
    params = ModelParameters(
        n_units=2_000,
        n_periods=120,
        n_replications=20,
        freeze_start_period=24,
        burn_in_periods=24,
    )
    output = ROOT / "output"
    paper_dir = ROOT / "paper"

    print("Step 1/4: Monte Carlo simulation (3 scenarios, parallel replications)...")
    scenarios = [
        PolicyScenario.BASELINE,
        PolicyScenario.RENT_FREEZE,
        PolicyScenario.PARTIAL_CAP,
    ]
    results = {}
    for i, scenario in enumerate(scenarios, 1):
        print(f"  [{i}/3] {scenario.value}", flush=True)
        config = SimulationConfig(parameters=params, scenario=scenario)
        results[scenario.value] = run_experiment(config, parallel=True)

    save_results(results, output)
    generate_all_figures(
        results,
        output / "figures",
        freeze_start=params.freeze_start_period,
        horizon=params.n_periods - 1,
    )

    terminal = params.n_periods - 1
    headline = export_paper_stats(results, paper_dir, terminal, params.freeze_start_period)
    print(f"  Exported {paper_dir / 'generated_macros.tex'}")

    print("\nStep 2/4: Compiling PDF...")
    for pass_num in (1, 2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "rent_freeze_simulation.tex"],
            cwd=paper_dir,
            capture_output=True,
            text=True,
        )
        log = (proc.stdout or "") + (proc.stderr or "")
        if "Output written on" not in log:
            print(log[-3000:])
            raise SystemExit(f"pdflatex pass {pass_num} failed")

    pdf_path = paper_dir / "rent_freeze_simulation.pdf"
    elapsed = time.perf_counter() - t0
    print(f"\nStep 3/4: Done in {elapsed:.0f}s")
    print(f"  PDF: {pdf_path}")
    print(
        f"  Headline effects: rent {headline['rent_drop_pct']:.1f}%, "
        f"landlord profit {headline['profit_drop_pct']:.1f}%, "
        f"quality delta {headline['quality_delta']:.3f}"
    )

    print("\nStep 4/4: Validating outputs...")
    import subprocess as sp

    proc = sp.run([sys.executable, str(ROOT / "scripts" / "validate.py")], cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit("Validation failed")


if __name__ == "__main__":
    main()
