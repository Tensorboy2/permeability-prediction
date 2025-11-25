'''
perm_vs_por_plot.py

A script for extracting the values of permeability plotted against the respective porosity values.
'''
import os
path = os.path.dirname(__file__)
data_path = os.path.join(path, 'data')


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib import gridspec, colors

def plot_perm_vs_por_v4(porosities, permeabilities, path=os.path.dirname(__file__)):
    """
    Plot permeability components over porosity values.
    - Two main subplots: one for diagonal (Kxx + Kyy) and one for off-diagonal (Kxy + Kyx)
    - Each main subplot has: 2D histogram (center), and marginal 1D histograms (top + right)
    - Shared color normalization for all 2D histograms.
    """
    porosity_values = 1 - porosities.mean(axis=(1, 2))
    cmap_2d = plt.cm.viridis
    cmap_1d = plt.cm.plasma

    # --- Merge diagonal and off-diagonal components ---
    diag_perm = np.concatenate([permeabilities[:, 0, 0], permeabilities[:, 1, 1]])
    offdiag_perm = np.concatenate([permeabilities[:, 0, 1], permeabilities[:, 1, 0]])
    poro_diag = np.concatenate([porosity_values, porosity_values])
    poro_offdiag = np.concatenate([porosity_values, porosity_values])

    # --- Compute normalization for 2D histograms ---
    h_diag = np.histogram2d(poro_diag, diag_perm, bins=100)[0]
    h_off = np.histogram2d(poro_offdiag, offdiag_perm, bins=100)[0]
    global_max_2d = np.max([h_diag.max(), h_off.max()])
    norm2d = colors.LogNorm(vmin=1, vmax=global_max_2d)

    def fit_C(p, K):
        mask = (p > 0) & (p < 1) & np.isfinite(K)
        C_vals = K[mask] * (1 - p[mask])**2 / (p[mask]**3)
        C_vals = C_vals[np.isfinite(C_vals) & (C_vals > 0)]
        if len(C_vals) == 0:
            return np.nan
        return np.median(C_vals)
    # --- Fit constants ---
    C_diag = fit_C(poro_diag, diag_perm)
    C_off = fit_C(poro_offdiag, offdiag_perm)

    # --- Compute normalization for 1D histograms ---
    N_poro_diag, _ = np.histogram(poro_diag, bins=100)
    N_poro_off, _ = np.histogram(poro_offdiag, bins=100)
    N_perm_diag, _ = np.histogram(diag_perm, bins=100)
    N_perm_off, _ = np.histogram(offdiag_perm, bins=100)
    global_max_1d = max(N_poro_diag.max(), N_poro_off.max(), N_perm_diag.max(), N_perm_off.max())
    norm1d = colors.LogNorm(vmin=1, vmax=global_max_1d)

    # --- Figure setup ---
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    fig = plt.figure(figsize=(7, 3.5))
    outer_gs = gridspec.GridSpec(1, 2, wspace=0.4, hspace=0.3)
    mappable2d = None  # for shared colorbar

    # --- Helper data structure ---
    data_groups = [
        ("($K_{xx}, K_{yy}$)", poro_diag, diag_perm, C_diag),
        ("($K_{xy}, K_{yx}$)", poro_offdiag, offdiag_perm, C_off)
    ]

    for col, (title, poro, perm,C_fit) in enumerate(data_groups):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer_gs[col],
            width_ratios=[4, 1], height_ratios=[1, 4],
            wspace=0.05, hspace=0.05
        )

        ax_histx = fig.add_subplot(inner_gs[0, 0])
        ax_scatter = fig.add_subplot(inner_gs[1, 0])
        ax_histy = fig.add_subplot(inner_gs[1, 1])

        # --- 2D histogram ---
        h = ax_scatter.hist2d(poro, perm, bins=100, cmap=cmap_2d, norm=norm2d, cmin=1)
        if mappable2d is None:
            mappable2d = h[3]
        ax_scatter.set_xlabel(f"Porosity")
        ax_scatter.set_ylabel(f"Permeability{title}")
        # ax_scatter.set_title(title, fontsize=12)
        ax_scatter.grid(True)

        if np.isfinite(C_fit) and col == 0:
            p_fit = np.linspace(poro.min(), poro.max(), 200)
            k_fit = C_fit * (p_fit**3) / ((1 - p_fit)**2)
            ax_scatter.plot(p_fit, k_fit, 'r-', lw=2, label=fr"$KC = {C_fit:.2e}\, \frac{{p^3}}{{(1-p)^2}}$")
            ax_scatter.legend(fontsize=8, loc='upper left')
        elif np.isfinite(C_fit) and col == 1:
            p_fit = np.linspace(poro.min(), poro.max(), 200)
            # k_fit = C_fit * (p_fit**3) / ((1 - p_fit)**2)
            k_fit = np.zeros_like(p_fit)
            ax_scatter.plot(p_fit, k_fit, 'r-', lw=2, label=fr"$KC = 0$")
            ax_scatter.legend(fontsize=8, loc='upper left')

        # --- Top histogram (porosity) ---
        N, bins, _ = ax_histx.hist(poro, bins=100, color='black', alpha=0.9)
        ax_histx.set_xticks([])
        ax_histx.set_xlim(ax_scatter.get_xlim())

        # --- Right histogram (permeability) ---
        N, bins, _ = ax_histy.hist(perm, bins=100, color='black', alpha=0.9, orientation='horizontal')
        ax_histy.set_yticks([])
        ax_histy.set_ylim(ax_scatter.get_ylim())

    # --- Shared colorbar ---
    cbar2d = fig.colorbar(
        mappable2d, ax=fig.get_axes(),
        orientation='vertical', fraction=0.10, pad=0.02
    )
    cbar2d.set_label("Counts (2D histograms)")

    plt.savefig(os.path.join(path, 'perm_vs_porosity_v5.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)



if __name__ == "__main__":

    # Fetch perm and porosity values from the data directory
    train_and_val_images = np.load(os.path.join(data_path, 'images_filled.npy'),mmap_mode='r')
    permeabilities = np.load(os.path.join(data_path, 'train_and_val_k.npy'),mmap_mode='r')

    test_images = np.load(os.path.join(data_path, 'test_images_filled.npy'),mmap_mode='r')
    test_permeabilities = np.load(os.path.join(data_path, 'test_k.npy'),mmap_mode='r')
    permeabilities = np.concatenate((permeabilities, test_permeabilities), axis=0)

    # print(porosities.shape, permeabilities.shape)
    # plot_perm_vs_por_v1(porosities, permeabilities)
    all_images = np.concatenate([train_and_val_images, test_images], axis=0)

    plot_perm_vs_por_v4(all_images, permeabilities)