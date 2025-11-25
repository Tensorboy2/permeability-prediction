'''
percolation.py

This module contains the functionality to generate the images and check the percolation.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label, binary_dilation
from matplotlib.colors import ListedColormap, BoundaryNorm
import os

# Constants
STRUCTURE_8 = np.ones((3, 3), dtype=int)
SCRIPT_DIR = os.path.dirname(__file__)


def generate_periodic_blobs(shape=(128, 128), density=0.5, sigma=2.0, seed=0):
    '''
    Modified from the scipy.ndimage.binary_blobs function to include mode "wrap", for periodic wrapping on structures.
    '''
    rng = np.random.default_rng(seed)
    noise = rng.random(shape)
    smooth = gaussian_filter(noise, sigma=sigma, mode='wrap')
    threshold = np.percentile(smooth, (1.0 - density) * 100)
    return smooth > threshold  # True = solid, False = fluid

def generate_cross_channels(shape=(64, 64), width=3):
    img = np.ones(shape, dtype=bool)  # True = solid
    h, w = shape

    for i in range(h // 2):
        x1, y1 = i, h // 2 - i - 1
        x2, y2 = w - h // 2 + i, h - i - 1

        if 0 <= y1 < h and 0 <= x1 < w:
            img[max(y1 - width // 2, 0):y1 + width // 2 + 1,
                max(x1 - width // 2, 0):x1 + width // 2 + 1] = False

        if 0 <= y2 < h and 0 <= x2 < w:
            img[max(y2 - width // 2, 0):y2 + width // 2 + 1,
                max(x2 - width // 2, 0):x2 + width // 2 + 1] = False

    return img


def label_fluid_regions(binary_img):
    """Labels fluid regions where img == False."""
    return label(~binary_img, structure=STRUCTURE_8)

def detect_percolation(binary_img):
    """Check if fluid clusters percolate in x or y (periodic)."""
    dilated = binary_dilation(binary_img)
    labeled, _ = label_fluid_regions(dilated)
    h, w = labeled.shape

    # Boundary check:
    x_perc = any(labeled[0, j] != 0 and labeled[0, j] == labeled[-1, j] for j in range(w))
    y_perc = any(labeled[i, 0] != 0 and labeled[i, 0] == labeled[i, -1] for i in range(h))

    return x_perc, y_perc, labeled

def fill_non_percolating_fluid(binary_img):
    """Turns non-percolating fluid regions into solid."""
    labels, _ = label_fluid_regions(binary_img)
    top, bottom = set(labels[0, :]) - {0}, set(labels[-1, :]) - {0}
    left, right = set(labels[:, 0]) - {0}, set(labels[:, -1]) - {0}

    percolating = (top & bottom) | (left & right)
    non_perc_mask = (labels > 0) & ~np.isin(labels, list(percolating))

    filled_img = binary_img.copy()
    filled_img[non_perc_mask] = True  # fill in fluid to solid
    return filled_img


def plot_percolation_results(original, labeled, filled, percolates_x, percolates_y, save_path=None):
    '''
    Plotting of the original image, the labeled fluid clusters and the filled image.
    '''
    num_labels = labeled.max()
    jet_colors = plt.cm.jet(np.linspace(0, 1, num_labels))
    colors = np.vstack(([1, 1, 1, 1], jet_colors))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, num_labels + 1.5), len(colors))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    titles = ["Original", "Clusters", f"Filled\nPercolates: x={percolates_x}, y={percolates_y}"]
    images = [original, labeled, filled]
    cmaps = ['gray', cmap, 'gray']
    norms = [None, norm, None]

    for ax, img, title, cmap_, norm_ in zip(axes, images, titles, cmaps, norms):
        ax.imshow(img, cmap=cmap_, norm=norm_)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    # Switch between blob or cross image here:
    img = generate_cross_channels(shape=(128, 128), width=30)
    # img = generate_periodic_blobs(shape=(128, 128), density=0.5, sigma=2.0, seed=9)

    px, py, labeled = detect_percolation(img)
    filled = fill_non_percolating_fluid(img)

    print("Percolates:", px and py)
    save_file = os.path.join(SCRIPT_DIR, "percolation_check_demo.pdf")
    plot_percolation_results(img, labeled, filled, px, py, save_path=save_file)

if __name__ == "__main__":
    main()
