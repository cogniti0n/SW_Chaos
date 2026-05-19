import matplotlib.pyplot as plt


def setup_matplotlib_for_paper(
    figsize=(6, 4),
    font_size=14,
    linewidth=2.0,
    axes_linewidth=1.5,
    tick_length=6,
    tick_width=1.5,
    tick_pad=7,
    marker_size=8.0,
    marker_edgewidth=1.5,
):
    params = {
        "figure.figsize": figsize,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "png",
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "lines.linewidth": linewidth,
        "lines.markersize": marker_size,
        "lines.markeredgewidth": marker_edgewidth,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "axes.linewidth": axes_linewidth,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.pad": tick_pad,
        "ytick.major.pad": tick_pad,
        "xtick.major.size": tick_length,
        "ytick.major.size": tick_length,
        "xtick.major.width": tick_width,
        "ytick.major.width": tick_width,
        "xtick.top": True,
        "ytick.right": True,
        "legend.fontsize": font_size,
        "legend.frameon": False,
    }
    plt.rcParams.update(params)
    print(f"Matplotlib style updated.")
