# Paper build

```bash
# From project root — full pipeline (~10 min)
python scripts/build_paper.py

# Or compile only (after simulation has run)
cd paper && pdflatex rent_freeze_simulation.tex
```

Numbers in `generated_macros.tex` and `generated_table.tex` are auto-written from simulation output. Do not edit by hand.

Figures are loaded from `../output/figures/`.
