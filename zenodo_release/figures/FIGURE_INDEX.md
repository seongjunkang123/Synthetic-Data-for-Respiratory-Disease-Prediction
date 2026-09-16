# Figure files and manuscript numbering

Stable filenames are retained so existing reproduction commands continue to work.
Each figure is supplied as a vector PDF, an editable SVG and a 300-dpi PNG.

| Manuscript number | File stem | Content |
|---|---|---|
| Figure 1 | fig1_pipeline | Dataset generation and patient-level evaluation |
| Figure 2 | fig3_marginals | Abundance distributions and KS distances |
| Figure 3 | fig4_correlation | Spearman correlation matrices |
| Figure 4 | fig5_pca | Real-data principal components and dispersion |
| Figure 5 | fig6_utility | Held-out classification and run variation |
| Figure 6 | fig7_baselines | Synthesis-method comparison |
| Figure 7 | fig8_privacy | Nearest-neighbour distances |
| Figure A1 | fig2_training | Archived generator training history |

## Visual conventions

All figures use DejaVu Sans and a white background. Real profiles use charcoal
(#263238) and synthetic profiles use muted purple (#7A5195) when comparing record
sources. Diagnostic colours are asthma blue (#0072B2), bronchiectasis green
(#009E73) and COPD vermilion (#D55E00). Open and filled points, labelled axes and
line styles supplement colour. Correlation matrices use a common blue-white-brown
diverging map, with symmetric limits shown on each colour bar.

The workflow explicitly routes held-out real patients around model fitting.
Figure styling does not change observations, sample selection, plotted scales,
summary statistics or benchmark results.

Reproduce with code/figures_schematic.py, code/figures_fidelity.py and
code/figures_utility.py. The common style and export settings are in
code/figure_style.py.
