import numpy as np
import matplotlib.pyplot as plt
import os
path = os.path.dirname(__file__)

from percolation import find_percolating_clusters, fill_non_percolating, periodic_binary_blobs

def porous_media_generation(num_img=1, img_size=128, porosity_range=[0.1,0.5], sigma=4.0):
    """
    ## Generate set of porous media.

    This function generates a set of 2D porous media based porosity and image size. 
    The media is defined to be periodic in both axis and it ensures percolation in both directions.

    ## Params:
    - num_img (int) The number of images for generation
    - img_size (int) The size of the image 
    - img_size ([float,float]) The size of the image 

    """
    images = np.zeros((num_img,img_size,img_size),dtype=np.int32)
    images_filled = np.zeros((num_img,img_size,img_size),dtype=np.int32)
    porosities = np.zeros(num_images)
    num_images_made=0
    i=0
    while (i<num_img):
        porosity = porosity_range[0] + np.random.rand()*(porosity_range[1] - porosity_range[0])
        img = periodic_binary_blobs(shape=(img_size,img_size), blob_density=1-porosity, sigma=sigma,seed=num_images_made)
        px, py, labeled = find_percolating_clusters(img)
        percolates = px and py
        if percolates:
            filled = fill_non_percolating(img)
            images[i] = img
            images_filled[i] = filled
            porosities[i] = porosity
            i+=1
        num_images_made+=1
        print(f"Number of images made: {num_images_made}, kept: {i}")
    np.save(os.path.join(path,"images.npy"),images)
    np.save(os.path.join(path,"images_filled.npy"),images_filled)
    np.save(os.path.join(path,"porosities.npy"),porosities)
    


def compute_porosity(blobs):
    # pore space is False
    return 1-np.sum(blobs)/blobs.shape[-1]**2

def demo():
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # Parameter grid
    sigmas = [2.0, 4.0, 6.0]
    densities = [0.1, 0.3, 0.5]

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    # Generate subplots
    for i, sigma in enumerate(sigmas):
        for j, density in enumerate(densities):
            ax = axes[i, j]
            blobs = periodic_binary_blobs(shape=(128, 128), 
                                            blob_density=density, 
                                            sigma=sigma,seed=9)
            p_x, p_y, labeled = find_percolating_clusters(blobs)
            num_labels = labeled.max()
            jet_colors = plt.cm.jet(np.linspace(0, 1, num_labels))
            colors = np.vstack(([1, 1, 1, 1], jet_colors))  # white for label 0 (solid region)
            custom_cmap = ListedColormap(colors)
            norm = BoundaryNorm(np.arange(-0.5, num_labels + 1.5), len(colors))

            # blobs = make_img(img_size=128, blob_density=density, sigma=sigma,seed = 42)
            filled = fill_non_percolating(blobs)
            porosity = compute_porosity(filled)
            # ax.imshow(labeled, cmap=custom_cmap, norm=norm)
            ax.imshow(filled, cmap='gray')
            ax.set_title(f"sigma={sigma}, blob_density={density}\n porosity={porosity:.2f}, px: {p_x} py: {p_y}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(path,"blob_density_and_sigma_variation_demo.pdf"))
    plt.clf()
if __name__ == "__main__":
    # demo()
    num_images = 20_000
    img_size = 128
    porosity_range = [0.001,0.9]
    porous_media_generation(num_img=num_images,
                            img_size=img_size,
                            porosity_range=porosity_range)
    
    
