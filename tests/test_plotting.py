import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdakit import plotting


def _collect_axes(fig):
    axes = fig.axes.copy()
    plt.close(fig)
    return axes


def test_heat_hist2d_default_cmap_and_layout():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    fig, ax_main, ax_top, ax_right = plotting.heat_hist2d(x, y)
    axes = _collect_axes(fig)
    # Main + marginals + colorbar axes expected
    assert len(axes) == 4
    assert ax_main in axes and ax_top in axes and ax_right in axes
    # The image colormap should match the registered free-energy colormap
    mappables = [artist for artist in ax_main.get_children() if hasattr(artist, "get_cmap")]
    assert any(artist.get_cmap().name == plotting.FREE_ENERGY_CMAP.name for artist in mappables)


def test_heat_hist2d_from_proj_matches_direct_call():
    proj = np.arange(200, dtype=float).reshape(100, 2)
    fig_proj, *_ = plotting.heat_hist2d_from_proj(proj, i=0, j=1, bins=10)
    fig_direct, *_ = plotting.heat_hist2d(proj[:, 0], proj[:, 1], bins=10)
    # Compare the histogram arrays produced by imshow; stored in the image data
    img_proj = fig_proj.axes[0].images[0]
    img_direct = fig_direct.axes[0].images[0]
    np.testing.assert_allclose(img_proj.get_array(), img_direct.get_array())
    plt.close(fig_proj)
    plt.close(fig_direct)


def test_eigen_spectrum_returns_axes_with_titles():
    vals = np.array([3.0, 1.5, 0.5])
    fig, ax = plotting.eigen_spectrum(vals, logy=True, title="Spectrum")
    assert ax.get_title() == "Spectrum"
    assert ax.get_xscale() == "linear"
    assert ax.get_yscale() == "log"
    assert ax.get_lines()
    _collect_axes(fig)


def test_timescales_bar_handles_infinite_values():
    times = np.array([np.inf, 2.0, 1.0])
    fig, ax = plotting.timescales_bar(times, title="Times")
    heights = [bar.get_height() for bar in ax.patches]
    assert heights[0] == max(heights)
    assert ax.get_title() == "Times"
    _collect_axes(fig)


def test_scatter2d_from_proj_creates_points():
    proj = np.array([[0.0, 1.0], [1.0, 0.5], [2.0, 0.0]])
    fig, ax = plotting.scatter2d_from_proj(proj, i=0, j=1, title="Scatter")
    assert ax.collections  # scatter returns PathCollection
    assert ax.get_title() == "Scatter"
    _collect_axes(fig)


def test_component_hist_from_proj_density_histogram():
    proj = np.linspace(0, 1, 100)[:, None]
    fig, ax = plotting.component_hist_from_proj(proj, i=0, bins=10, density=True)
    assert ax.patches
    _collect_axes(fig)


def test_corner_from_proj_axes_grid():
    proj = np.random.default_rng(1).normal(size=(50, 3))
    fig, axes = plotting.corner_from_proj(proj, ks=(0, 1, 2), bins=5)
    axes = np.array(axes)
    assert axes.shape == (3, 3)
    _collect_axes(fig)
