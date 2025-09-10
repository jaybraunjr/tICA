import numpy as np
import matplotlib.pyplot as plt


def heat_hist2d(x, y, bins=100, range=None, cmap="viridis", figsize=(6, 6),
                top_frac=0.2, right_frac=0.2, wspace=0.05, hspace=0.05,
                x_label="IC 1", y_label="IC 2", title=None, density=True,
                hist_kwargs=None, imshow_kwargs=None):
    """Plot a 2D heatmap with marginal histograms for x and y.

    Parameters
    - x, y: array-like
      1D arrays of equal length.
    - bins: int or [int, int]
      Number of bins for 2D hist and marginals.
    - range: ((xmin, xmax), (ymin, ymax)) | None
      Ranges for the histograms; if None, computed from data.
    - cmap: str or Colormap
      Colormap for heatmap.
    - figsize: (w, h)
      Figure size in inches.
    - top_frac, right_frac: float
      Fraction of figure height/width for the top/right histograms.
    - wspace, hspace: float
      Spacing between axes.
    - x_label, y_label: str
      Axis labels.
    - title: str | None
      Figure title.
    - density: bool
      If True, normalize histograms.
    - hist_kwargs: dict | None
      Extra kwargs for the marginal histograms.
    - imshow_kwargs: dict | None
      Extra kwargs for the heatmap.

    Returns
    - fig, ax_main, ax_top, ax_right
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    hist_kwargs = {} if hist_kwargs is None else dict(hist_kwargs)
    imshow_kwargs = {} if imshow_kwargs is None else dict(imshow_kwargs)

    # Layout via GridSpec-like manual placement
    fig = plt.figure(figsize=figsize)
    # Compute axis rectangles
    left = 0.1
    bottom = 0.1
    right = 0.95
    top = 0.95
    width = right - left
    height = top - bottom

    main_w = width * (1 - right_frac)
    main_h = height * (1 - top_frac)
    ax_main = fig.add_axes([left, bottom, main_w, main_h])
    ax_top = fig.add_axes([left, bottom + main_h + hspace, main_w, height * top_frac - hspace])
    ax_right = fig.add_axes([left + main_w + wspace, bottom, width * right_frac - wspace, main_h])

    # 2D histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range, density=density)
    # Transpose for imshow (y first dimension)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax_main.imshow(H.T, origin='lower', extent=extent, aspect='auto', cmap=cmap, **imshow_kwargs)
    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)
    if title:
        ax_main.set_title(title)
    # Marginals
    ax_top.hist(x, bins=bins if np.isscalar(bins) else bins[0], range=None if range is None else range[0],
                density=density, color='gray', **hist_kwargs)
    ax_right.hist(y, bins=bins if np.isscalar(bins) else bins[1], range=None if range is None else range[1],
                  density=density, orientation='horizontal', color='gray', **hist_kwargs)

    # Ticks cleanup
    ax_top.set_xlim(extent[0], extent[1])
    ax_right.set_ylim(extent[2], extent[3])
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_right.tick_params(axis='y', labelleft=False)

    # Colorbar
    cax = fig.add_axes([left + main_w + wspace, bottom + main_h * 0.7, width * right_frac - wspace, main_h * 0.3])
    fig.colorbar(im, cax=cax)

    return fig, ax_main, ax_top, ax_right


def heat_hist2d_from_proj(proj, i=0, j=1, **kwargs):
    """Convenience wrapper to plot from a projection matrix (T x k).

    Parameters
    - proj: ndarray (T, k)
    - i, j: component indices for x/y
    - kwargs: forwarded to heat_hist2d
    """
    x = proj[:, i]
    y = proj[:, j]
    return heat_hist2d(x, y, x_label=f"IC {i+1}", y_label=f"IC {j+1}", **kwargs)


def eigen_spectrum(eigvals, logy=False, title="Eigenvalues", figsize=(6,4)):
    """Line plot of eigenvalues (scree plot).

    Parameters
    - eigvals: 1D array
    - logy: bool, plot y-axis on log scale
    - title: str
    - figsize: tuple
    """
    v = np.asarray(eigvals)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(1, v.size+1), v, 'o-', lw=1)
    ax.set_xlabel('Component')
    ax.set_ylabel('Eigenvalue')
    if logy:
        ax.set_yscale('log')
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def timescales_bar(timescales, title="Implied Timescales", figsize=(6,4), logy=True):
    """Bar plot of implied timescales.

    Parameters
    - timescales: 1D array (inf values are clipped at max finite)
    - logy: bool, use log scale for y
    """
    t = np.asarray(timescales)
    finite = np.isfinite(t)
    ymax = t[finite].max() if finite.any() else 1.0
    t_plot = t.copy()
    t_plot[~finite] = ymax
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(1, t.size+1), t_plot, color='tab:blue')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('Component')
    ax.set_ylabel('Timescale')
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def scatter2d_from_proj(proj, i=0, j=1, s=8, alpha=0.5, figsize=(6,5), title=None):
    """Simple 2D scatter of two components from a projection matrix (T x k)."""
    x = proj[:, i]
    y = proj[:, j]
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=s, alpha=alpha)
    ax.set_xlabel(f'IC {i+1}')
    ax.set_ylabel(f'IC {j+1}')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def component_hist_from_proj(proj, i=0, bins=100, density=True, figsize=(6,4), title=None):
    """1D histogram of a single component from projection matrix."""
    x = proj[:, i]
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(x, bins=bins, density=density, color='gray', edgecolor='none')
    ax.set_xlabel(f'IC {i+1}')
    ax.set_ylabel('Density' if density else 'Count')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def corner_from_proj(proj, ks=(0,1,2), bins=60, s=4, alpha=0.5, figsize=None):
    """Corner-like grid of pairwise scatters with marginals on diagonal.

    Parameters
    - proj: ndarray (T, k)
    - ks: iterable of component indices to include
    - bins: int, bins for diagonal histograms
    - s, alpha: scatter style
    - figsize: optional manual figsize
    """
    ks = list(ks)
    d = len(ks)
    if figsize is None:
        figsize = (3*d, 3*d)
    fig, axes = plt.subplots(d, d, figsize=figsize)
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.hist(proj[:, ks[i]], bins=bins, color='gray', edgecolor='none')
                ax.set_ylabel('')
            elif i > j:
                ax.scatter(proj[:, ks[j]], proj[:, ks[i]], s=s, alpha=alpha)
            else:
                ax.axis('off')
            if i == d-1 and j < d:
                ax.set_xlabel(f'IC {ks[j]+1}')
            else:
                ax.tick_params(axis='x', labelbottom=False)
            if j == 0 and i < d:
                ax.set_ylabel(f'IC {ks[i]+1}')
            else:
                ax.tick_params(axis='y', labelleft=False)
    fig.tight_layout()
    return fig, axes
