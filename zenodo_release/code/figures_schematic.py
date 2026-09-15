"""Figure 1: separate the full-cohort release and held-out benchmark workflows."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import FIGURES

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7.1, 4.3))
ax.set_xlim(-1, 101); ax.set_ylim(0, 99); ax.axis('off')
def box(x, y, w, h, title, lines, color='#edf3f8'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4',
                              fc=color, ec='#36516b', lw=.9))
    ax.text(x+w/2, y+h-3, title, ha='center', va='top', fontsize=8.2, weight='bold')
    ax.text(x+w/2, y+h-9, '\n'.join(lines), ha='center', va='top', fontsize=7.5,
            linespacing=1.5)
def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=11, lw=1, color='#36516b'))
box(5, 80, 90, 16, 'Published source and harmonised matrix',
    ['121 patients | 53 asthma, 35 bronchiectasis, 33 COPD | 78 shared PubChem CIDs'])
ax.text(5, 67, 'Release generation: fit on the full source matrix', fontsize=9, weight='bold')
box(5, 42, 27, 22, 'Fit and archive', ['Saved log1p / min-max', 'Conditional WGAN-GP'])
box(37, 42, 26, 22, 'Generate profiles', ['Seeded model draws', 'Inverse transform'])
box(68, 42, 27, 22, 'Reference dataset', ['6,000 synthetic records', '2,000 per class'])
arrow((32.5, 53), (36.5, 53)); arrow((63.5, 53), (67.5, 53))
ax.text(5, 33, 'Benchmark evaluation: refit within each real training fold', fontsize=9, weight='bold')
box(5, 8, 27, 22, 'Define patient folds', ['Five fixed splits', 'Training-only transforms'], '#f0edf7')
box(37, 8, 26, 22, 'Fit synthesis methods', ['Generate within folds', '500 draws per class'], '#f0edf7')
box(68, 8, 27, 22, 'Evaluate and archive', ['Untouched real test folds', 'Samples and predictions'], '#f0edf7')
arrow((32.5, 19), (36.5, 19)); arrow((63.5, 19), (67.5, 19))
ax.plot([5, 1, 1], [88, 88, 19], c='#36516b', lw=.9)
arrow((1, 53), (4.5, 53))
arrow((1, 19), (4.5, 19))
fig.savefig(FIGURES/'fig1_pipeline.pdf', bbox_inches='tight')
plt.close(fig)
