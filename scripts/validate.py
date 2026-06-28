#!/usr/bin/env python3
"""Verify simulation outputs, generated LaTeX stats, figures, and PDF are consistent."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TERMINAL = 119
TOLERANCE = 0.015  # allow rounding in LaTeX table (2 decimal places)


def _parse_table_row(line: str) -> dict[str, float]:
    parts = [p.strip() for p in line.split("&")]
    return {
        "mean_rent": float(parts[1]),
        "mean_quality": float(parts[2]),
        "vacancy_rate": float(parts[3]),
        "mean_tenant_utility": float(parts[4]),
        "mean_landlord_profit": float(parts[5].split("\\\\")[0]),
    }


def main() -> None:
    errors: list[str] = []

    # --- CSV summaries exist ---
    for scenario in ("baseline", "rent_freeze", "partial_cap"):
        path = ROOT / "output" / f"summary_{scenario}.csv"
        if not path.exists():
            errors.append(f"Missing {path}")

    # --- Figures ---
    for fig in ("fig_dynamics.pdf", "fig_welfare.pdf", "fig_supply.pdf"):
        path = ROOT / "output" / "figures" / fig
        if not path.exists() or path.stat().st_size < 5_000:
            errors.append(f"Missing or empty figure: {path}")

    # --- PDF ---
    pdf = ROOT / "paper" / "rent_freeze_simulation.pdf"
    if not pdf.exists() or pdf.stat().st_size < 50_000:
        errors.append(f"Missing or corrupt PDF: {pdf}")

    # --- Generated LaTeX matches CSV at terminal period ---
    table_path = ROOT / "paper" / "generated_table.tex"
    if table_path.exists():
        rows = {
            "Baseline": "baseline",
            "Hard freeze": "rent_freeze",
            "Vacancy decontrol": "partial_cap",
        }
        for line in table_path.read_text(encoding="utf-8").splitlines():
            if "&" not in line or line.startswith("%"):
                continue
            label = line.split("&")[0].strip()
            if label not in rows:
                continue
            tex_vals = _parse_table_row(line)
            csv = pd.read_csv(ROOT / "output" / f"summary_{rows[label]}.csv")
            csv_row = csv.loc[csv["period"] == TERMINAL].iloc[0]
            for key in tex_vals:
                diff = abs(tex_vals[key] - csv_row[key])
                if diff > TOLERANCE:
                    errors.append(
                        f"{label}.{key}: LaTeX={tex_vals[key]}, CSV={csv_row[key]:.4f}, diff={diff:.4f}"
                    )

    # --- Macros file present ---
    macros = ROOT / "paper" / "generated_macros.tex"
    if not macros.exists():
        errors.append("Missing generated_macros.tex")
    else:
        text = macros.read_text(encoding="utf-8")
        for cmd in ("SimPolicyStart", "SimTerminalPeriod", "SimRentDropPct"):
            if f"\\newcommand{{\\{cmd}}}" not in text:
                errors.append(f"Missing LaTeX macro: {cmd}")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("All validation checks passed.")
    print(f"  PDF: {pdf} ({pdf.stat().st_size:,} bytes)")
    print(f"  Terminal period: {TERMINAL}")
    print(f"  Figures: {ROOT / 'output' / 'figures'}")


if __name__ == "__main__":
    main()
