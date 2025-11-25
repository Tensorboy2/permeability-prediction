'''
demo_periodic_media.py
Demonstration of the generation of periodic porous media with percolation in both directions.
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from percolation import generate_periodic_blobs, detect_percolation, fill_non_percolating_fluid
from lbm import big_LBM

path = os.path.dirname(__file__)

def demo():
    # Parameters
    sigma = 4.0
    porosity = 0.4
    
    # Generate porous medium
    for i in range(1000):
        img = generate_periodic_blobs(shape=(128, 128), density=porosity, sigma=sigma, seed=i)
        px, py, labeled = detect_percolation(img)
        percolates = px and py
        if not percolates:
            print(f"Seed {i} did not produce percolating structure, retrying...")
            continue
        else:
            # img = fill_non_percolating_fluid(img)
            print(f"Seed {i} produced percolating structure.")
            break
    h, w = img.shape
    
    # Run LBM
    u, _, _ = big_LBM(img, T=10000, force_dir=1)
    u_mag = np.sqrt(u[..., 0]**2 + u[..., 1]**2)
    avg_u = np.max(u_mag[img == 0])
    u_scaled = u_mag / avg_u
    
    # Tile everything
    tiled_img = np.tile(img, (3, 3))
    tiled_u = np.tile(u_scaled, (3, 3))
    
    # Mask velocity where there are solids
    tiled_u_masked = np.ma.masked_where(tiled_img == 1, tiled_u)
    
    # Plot setup
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    
    # Background - show structure faintly everywhere
    bg = np.ones((3*h, 3*w, 4))
    bg[tiled_img == 1, :3] = 0.7  # Gray solids
    bg[tiled_img == 0, :3] = 0.95  # Near-white pores
    bg[..., 3] = 0.3  # Faint everywhere
    ax.imshow(bg, interpolation="none", extent=(0, 3*w, 3*h, 0))
    
    # Velocity field in pores only - faded in surrounding tiles
    velocity_alpha = np.zeros((3*h, 3*w))
    velocity_alpha[h:2*h, w:2*w] = 1.0  # Full opacity in center
    velocity_alpha[:h, :] = 0.2  # Faded elsewhere
    velocity_alpha[2*h:, :] = 0.2
    velocity_alpha[:, :w] = 0.2
    velocity_alpha[:, 2*w:] = 0.2
    velocity_alpha[tiled_img == 1] = 0  # No velocity in solids
    
    cmap = plt.cm.inferno
    norm = plt.cm.colors.Normalize(vmin=np.percentile(u_scaled[img==0], 1), 
                                     vmax=np.percentile(u_scaled[img==0], 99))
    velocity_rgba = cmap(norm(tiled_u))
    velocity_rgba[..., 3] = velocity_alpha
    
    ax.imshow(velocity_rgba, interpolation="none", extent=(0, 3*w, 3*h, 0))
    
    # Central domain - solid structure fully opaque
    center_overlay = np.ones((h, w, 4))
    center_overlay[img == 0, 3] = 0  # Transparent in pores
    center_overlay[img == 1, :3] = 0.95  # White solids
    center_overlay[img == 1, 3] = 1.0  # Fully opaque
    ax.imshow(center_overlay, extent=(w, 2*w, 2*h, h), interpolation="none")
    
    # Frame
    # rect = plt.Rectangle(
    #     (w, h), w, h, linewidth=2.5, edgecolor='black', 
    #     facecolor='none', alpha=0.5, linestyle='-'
    # )
    # ax.add_patch(rect)
    
    # View limits
    margin = w // 5
    ax.set_xlim(w - margin, 2*w + margin)
    ax.set_ylim(2*h + margin, h - margin)
    ax.axis('off')
    
    # Colorbar
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$u / u_{max}$')
    # cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(path, "demo_periodic_media.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    plt.figure(figsize=(3.5, 3.5))
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(path, "demo_media.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    demo()