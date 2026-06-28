# Rent Freeze Agent-Based Simulation

Agent-based model of rental housing under rent freeze policy: heterogeneous landlords and tenants, search frictions, endogenous maintenance, Monte Carlo counterfactuals, and an auto-generated LaTeX paper.

## Final deliverable

| Artifact | Path |
|----------|------|
| **Paper PDF** | `paper/rent_freeze_simulation.pdf` |
| **LaTeX source** | `paper/rent_freeze_simulation.tex` |
| **Auto-generated stats** | `paper/generated_macros.tex`, `paper/generated_table.tex` |
| **Figures** | `output/figures/*.pdf` |
| **Simulation CSV** | `output/summary_*.csv` |

## One-command build (recommended)

```bash
pip install -r requirements.txt
python scripts/build_paper.py
```

Runs: simulation (2,000 units × 120 months × 20 replications) → figures → LaTeX export → PDF → validation.

## Verify everything

```bash
python tests/test_sanity.py     # model invariants
python scripts/validate.py      # CSV ↔ LaTeX ↔ figures ↔ PDF
```

## Quick smoke test (~1 min)

```bash
python scripts/run_simulation.py --fast
```

## Model summary

- **Tenants:** $U=\ln(y-r)+\theta\ln q$ with affordability cap (45% of income)
- **Landlords:** profit-max maintenance; quality $q'=(1-\rho)q+m^\gamma$; shadow-rent upkeep cut under cap
- **Policies:** baseline, hard freeze ($\bar r=0.88$ at month 24), vacancy decontrol
- **References:** Arnott (1995), Glaeser & Luttmer (2003), Sims & Schmitz (2011), Diamond et al. (2019)

## Project layout

```
src/model/           agents, market, policy, parameters
src/simulation.py    parallel Monte Carlo
src/visualization.py publication figures
src/paper_export.py  LaTeX stat export
scripts/build_paper.py
scripts/validate.py
tests/test_sanity.py
paper/
```
