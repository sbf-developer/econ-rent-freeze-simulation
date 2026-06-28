"""Export simulation results into LaTeX fragments for the paper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .simulation import ExperimentResult

SCENARIO_ROWS = [
    ("baseline", "Baseline"),
    ("rent_freeze", "Hard freeze"),
    ("partial_cap", "Vacancy decontrol"),
]


def terminal_row(result: ExperimentResult, period: int) -> pd.Series:
    row = result.summary.loc[result.summary["period"] == period]
    if row.empty:
        raise ValueError(f"No data for period {period}")
    return row.iloc[0]


def export_paper_stats(
    results: dict[str, ExperimentResult],
    paper_dir: Path,
    terminal_period: int,
    policy_start: int = 24,
) -> dict[str, float]:
    """Write LaTeX fragments; numbers in the paper always match simulation output."""
    base = terminal_row(results["baseline"], terminal_period)
    freeze = terminal_row(results["rent_freeze"], terminal_period)

    rent_drop_pct = 100 * (freeze["mean_rent"] - base["mean_rent"]) / base["mean_rent"]
    profit_drop_pct = 100 * (
        (freeze["mean_landlord_profit"] - base["mean_landlord_profit"])
        / base["mean_landlord_profit"]
    )
    quality_delta = freeze["mean_quality"] - base["mean_quality"]

    macro_lines = [
        "% Auto-generated — do not edit by hand",
        f"\\newcommand{{\\SimTerminalPeriod}}{{{terminal_period}}}",
        f"\\newcommand{{\\SimPolicyStart}}{{{policy_start}}}",
        f"\\newcommand{{\\SimRentDropPct}}{{{abs(rent_drop_pct):.1f}}}",
        f"\\newcommand{{\\SimProfitDropPct}}{{{abs(profit_drop_pct):.1f}}}",
        f"\\newcommand{{\\SimBaselineRent}}{{{base['mean_rent']:.2f}}}",
        f"\\newcommand{{\\SimFreezeRent}}{{{freeze['mean_rent']:.2f}}}",
        f"\\newcommand{{\\SimQualityDelta}}{{{quality_delta:.3f}}}",
        f"\\newcommand{{\\SimBaselineUtility}}{{{base['mean_tenant_utility']:.2f}}}",
        f"\\newcommand{{\\SimFreezeUtility}}{{{freeze['mean_tenant_utility']:.2f}}}",
        f"\\newcommand{{\\SimDecontrolUtility}}{{{terminal_row(results['partial_cap'], terminal_period)['mean_tenant_utility']:.2f}}}",
    ]

    table_lines = ["% Auto-generated table rows — do not edit by hand"]
    for key, label in SCENARIO_ROWS:
        row = terminal_row(results[key], terminal_period)
        table_lines.append(
            f"{label} & {row['mean_rent']:.2f} & {row['mean_quality']:.3f} & "
            f"{row['vacancy_rate']:.3f} & {row['mean_tenant_utility']:.2f} & "
            f"{row['mean_landlord_profit']:.2f} \\\\"
        )

    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "generated_macros.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")
    (paper_dir / "generated_table.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    return {
        "rent_drop_pct": rent_drop_pct,
        "profit_drop_pct": profit_drop_pct,
        "quality_delta": quality_delta,
    }
