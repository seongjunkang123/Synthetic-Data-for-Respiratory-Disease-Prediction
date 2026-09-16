"""Figure 1: full-cohort generation and evaluation with held-out patients."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import FIGURES
from figure_style import apply_style, save_figure

apply_style()
FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7.1, 4.5))
ax.set_xlim(-1, 101)
ax.set_ylim(0, 105)
ax.axis('off')

def box(x, y, w, h, title, lines, color='#EEF4F8'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.5,rounding_size=1.2',
                              fc=color, ec='#637783', lw=.8))
    ax.text(x+w/2, y+h-3, title, ha='center', va='top', fontsize=8.2, weight='bold')
    ax.text(x+w/2, y+h-9.5, '\n'.join(lines), ha='center', va='top', fontsize=7.8,
            linespacing=1.5)

def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=10,
                                lw=1, color='#526873'))

box(5, 87, 90, 15, 'Published source and harmonised matrix',
    ['121 patients  •  53 asthma, 35 bronchiectasis, 33 COPD  •  78 shared VOCs'])
ax.text(5, 76, 'a  Release generation', fontsize=9, weight='bold')
box(5, 50, 27, 22, 'Fit and archive', ['All 121 source profiles', 'WGAN-GP + transform'])
box(37, 50, 26, 22, 'Generate profiles', ['Seeded conditional draws', 'Inverse transform'])
box(68, 50, 27, 22, 'Reference dataset', ['6,000 synthetic profiles', '2,000 per diagnosis'])
arrow((32.5, 61), (36.5, 61))
arrow((63.5, 61), (67.5, 61))
ax.text(5, 40, 'b  Patient-level benchmark', fontsize=9, weight='bold')
box(5, 14, 27, 22, 'Define five folds', ['96–97 training patients', '24–25 test patients'], '#F3EFF6')
box(37, 14, 26, 22, 'Fit synthesis methods', ['Training patients only', '500 draws per diagnosis'], '#F3EFF6')
box(68, 14, 27, 22, 'Evaluate and archive', ['Held-out real testing', 'Samples + predictions'], '#F3EFF6')
arrow((32.5, 25), (36.5, 25))
arrow((63.5, 25), (67.5, 25))
# Source data feed the two procedures independently.
ax.plot([5, 1, 1], [94, 94, 25], c='#526873', lw=.9)
arrow((1, 61), (4.5, 61))
arrow((1, 25), (4.5, 25))
# The real test partition bypasses synthesis and all training transforms.
ax.plot([18.5, 18.5, 81.5], [13.5, 5, 5], c='#526873', lw=.9)
arrow((81.5, 5), (81.5, 13.5))
ax.text(50, 6.8, 'Held-out patients bypass model fitting', ha='center', va='bottom',
        fontsize=7.8, bbox=dict(facecolor='white', edgecolor='none', pad=1))
save_figure(fig, FIGURES/'fig1_pipeline.pdf')
plt.close(fig)
