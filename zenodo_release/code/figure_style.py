"""Shared publication style and vector/raster exports for all release figures."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DISEASE = {'asthma': '#0072B2', 'bronchiectasis': '#009E73', 'copd': '#D55E00'}
SYNTHETIC = '#7A5195'
INK = '#263238'
DIVERGING = LinearSegmentedColormap.from_list('breathnet_diverging', ['#2166AC', '#F7F7F7', '#A65D21'])

def apply_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2,
        'legend.fontsize': 7.2, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'lines.linewidth': 1,
        'xtick.top': False, 'ytick.right': False,
        'xtick.minor.visible': False, 'ytick.minor.visible': False,
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'mathtext.fontset': 'dejavusans', 'savefig.facecolor': 'white',
    })

def save_figure(fig, path):
    path = Path(path)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(path.with_suffix('.' + extension), bbox_inches='tight', dpi=300)
