"""
Publication-quality figures for rent-freeze simulation results.

Design principles:
  - Policy enactment line always matches freeze_start passed from simulation params.
  - Quality panel shows gap vs baseline (raw levels often overlap at the ceiling).
  - Misallocation uses level changes, not percent (baseline index is near zero).
  - Rental-stock panel uses percent change from initial capacity (readable scale).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .simulation import ExperimentResult

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

SCENARIO_LABELS = {
    "baseline": "Baseline (market)",
    "rent_freeze": "Hard rent freeze",
    "partial_cap": "Freeze + vacancy decontrol",
}

SCENARIO_COLORS = {
    "baseline": "#2166ac",
    "rent_freeze": "#b2182b",
    "partial_cap": "#4daf4a",
}

# Metrics where percent change vs baseline is meaningful at the terminal period.
PCT_METRICS = [
    ("mean_rent", "Mean rent", "Contract rent paid by matched tenants"),
    ("mean_quality", "Mean quality", "Physical condition index of rental units"),
    ("vacancy_rate", "Vacancy rate", "Share of units without a tenant"),
    ("mean_tenant_utility", "Tenant utility", "Average well-being of housed tenants"),
    ("mean_landlord_profit", "Landlord profit", "Average net operating income per unit"),
]


def _annotate_policy_start(ax: plt.Axes, freeze_start: int, show_label: bool = False) -> None:
    ax.axvline(freeze_start, color="0.35", linestyle="--", linewidth=1.2)
    if show_label:
        ax.text(
            freeze_start + 0.5,
            0.97,
            f"Policy start\n(month {freeze_start})",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
            color="0.25",
            va="top",
        )


def plot_time_series_panel(
    results: dict[str, ExperimentResult],
    freeze_start: int,
    output_path: Path,
) -> Path:
    """Four-panel dynamics: rent, quality gap, vacancy, tenant utility."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    baseline_df = results["baseline"].summary

    rent_ax, quality_ax, vacancy_ax, utility_ax = axes.flat

    # --- Rent ---
    for key, result in results.items():
        df = result.summary
        rent_ax.plot(
            df["period"],
            df["mean_rent"],
            label=SCENARIO_LABELS.get(key, key),
            color=SCENARIO_COLORS.get(key, "gray"),
            linewidth=1.8,
        )
    rent_ax.axhline(results["rent_freeze"].summary["mean_rent"].iloc[-1], color="0.7", ls=":", lw=1)
    _annotate_policy_start(rent_ax, freeze_start, show_label=True)
    rent_ax.set_title("Mean contract rent")
    rent_ax.set_xlabel("Month")
    rent_ax.set_ylabel("Rent (index; 1.0 ≈ pre-crisis median)")

    # --- Quality gap vs baseline (easier to read than overlapping levels) ---
    base_q = baseline_df.set_index("period")["mean_quality"]
    for key in ("rent_freeze", "partial_cap"):
        df = results[key].summary.set_index("period")
        gap = df["mean_quality"] - base_q
        quality_ax.plot(
            gap.index,
            gap.values,
            label=f"{SCENARIO_LABELS[key]} minus baseline",
            color=SCENARIO_COLORS[key],
            linewidth=1.8,
        )
    quality_ax.axhline(0.0, color="0.2", linewidth=0.8)
    _annotate_policy_start(quality_ax, freeze_start)
    quality_ax.set_title("Housing quality gap vs market baseline")
    quality_ax.set_xlabel("Month")
    quality_ax.set_ylabel("Δ Quality index")

    # --- Vacancy ---
    for key, result in results.items():
        df = result.summary
        vacancy_ax.plot(
            df["period"],
            100 * df["vacancy_rate"],
            label=SCENARIO_LABELS.get(key, key),
            color=SCENARIO_COLORS.get(key, "gray"),
            linewidth=1.8,
        )
    _annotate_policy_start(vacancy_ax, freeze_start)
    vacancy_ax.set_title("Vacancy rate")
    vacancy_ax.set_xlabel("Month")
    vacancy_ax.set_ylabel("Vacant units (%)")

    # --- Tenant utility ---
    for key, result in results.items():
        df = result.summary
        utility_ax.plot(
            df["period"],
            df["mean_tenant_utility"],
            label=SCENARIO_LABELS.get(key, key),
            color=SCENARIO_COLORS.get(key, "gray"),
            linewidth=1.8,
        )
    _annotate_policy_start(utility_ax, freeze_start)
    utility_ax.set_title("Mean tenant utility (matched households)")
    utility_ax.set_xlabel("Month")
    utility_ax.set_ylabel("Utility (log scale)")

    handles, labels = rent_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Dynamic Effects of a Rent Freeze (cross-replication averages)",
        y=1.07,
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_welfare_decomposition(
    results: dict[str, ExperimentResult],
    freeze_start: int,
    output_path: Path,
    horizon: Optional[int] = None,
) -> Path:
    """Percent change vs baseline; misallocation shown as absolute level change."""
    baseline = results["baseline"].summary
    terminal = horizon if horizon is not None else int(baseline["period"].max())

    fig, (ax_pct, ax_mis) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    metric_keys = [m[0] for m in PCT_METRICS]
    metric_labels = [m[1] for m in PCT_METRICS]
    x = np.arange(len(metric_keys))
    width = 0.35

    for i, (scenario_key, label) in enumerate(
        [("rent_freeze", "Hard freeze"), ("partial_cap", "Vacancy decontrol")]
    ):
        df = results[scenario_key].summary
        pct_changes = []
        for m in metric_keys:
            b_val = float(baseline.loc[baseline["period"] == terminal, m].values[0])
            s_val = float(df.loc[df["period"] == terminal, m].values[0])
            if abs(b_val) < 1e-8:
                pct_changes.append(0.0)
            else:
                pct_changes.append(100 * (s_val - b_val) / b_val)

        ax_pct.bar(
            x + i * width,
            pct_changes,
            width,
            label=label,
            color=SCENARIO_COLORS.get(scenario_key),
            alpha=0.9,
        )

    ax_pct.axhline(0, color="0.2", linewidth=0.8)
    ax_pct.set_xticks(x + width / 2)
    ax_pct.set_xticklabels(metric_labels, rotation=20, ha="right")
    ax_pct.set_ylabel(f"Percent change vs baseline\n(at month {terminal})")
    ax_pct.set_title("Terminal policy effects (%)")
    ax_pct.legend(loc="upper left", fontsize=8)

    # Misallocation: absolute change (baseline covariance is near zero)
    b_mis = float(baseline.loc[baseline["period"] == terminal, "misallocation_index"].values[0])
    mis_labels = []
    mis_vals = []
    for scenario_key, label in [("rent_freeze", "Hard freeze"), ("partial_cap", "Decontrol")]:
        s_mis = float(
            results[scenario_key].summary.loc[
                results[scenario_key].summary["period"] == terminal, "misallocation_index"
            ].values[0]
        )
        mis_labels.append(label)
        mis_vals.append(s_mis - b_mis)

    ax_mis.bar(mis_labels, mis_vals, color=[SCENARIO_COLORS["rent_freeze"], SCENARIO_COLORS["partial_cap"]])
    ax_mis.axhline(0, color="0.2", linewidth=0.8)
    ax_mis.set_title("Misallocation index\n(level change, not %)")
    ax_mis.set_ylabel("Δ Misallocation")

    fig.suptitle("Long-run counterfactual vs free market", y=1.02, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_landlord_profits_and_stock(
    results: dict[str, ExperimentResult],
    freeze_start: int,
    output_path: Path,
) -> Path:
    """Landlord profits and rental-stock change (percent from initial)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    for key, result in results.items():
        df = result.summary
        initial_stock = df["active_units"].iloc[0]
        stock_pct = 100 * (df["active_units"] / initial_stock - 1.0)
        ax1.plot(
            df["period"],
            df["mean_landlord_profit"],
            label=SCENARIO_LABELS.get(key, key),
            color=SCENARIO_COLORS.get(key, "gray"),
            linewidth=1.8,
        )
        ax2.plot(
            df["period"],
            stock_pct,
            label=SCENARIO_LABELS.get(key, key),
            color=SCENARIO_COLORS.get(key, "gray"),
            linewidth=1.8,
        )

    for ax in (ax1, ax2):
        _annotate_policy_start(ax, freeze_start)
    _annotate_policy_start(ax1, freeze_start, show_label=True)

    ax1.set_title("Landlord net operating income")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Profit (index)")
    ax1.legend(fontsize=8, loc="lower right")

    ax2.set_title("Rental stock change from initial capacity")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Change in active units (%)")
    ax2.axhline(0, color="0.2", linewidth=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all_figures(
    results: dict[str, ExperimentResult],
    output_dir: Path,
    freeze_start: int = 24,
    horizon: Optional[int] = None,
) -> list[Path]:
    if horizon is None:
        horizon = int(results["baseline"].summary["period"].max())

    return [
        plot_time_series_panel(results, freeze_start, output_dir / "fig_dynamics.png"),
        plot_welfare_decomposition(
            results, freeze_start, output_dir / "fig_welfare.png", horizon=horizon
        ),
        plot_landlord_profits_and_stock(results, freeze_start, output_dir / "fig_supply.png"),
    ]
