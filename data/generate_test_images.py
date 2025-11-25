import numpy as np
import matplotlib.pyplot as plt
import os
path = os.path.dirname(__file__)

from percolation import detect_percolation, fill_non_percolating_fluid, generate_periodic_blobs

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
    start_seed = 1_000_000
    i=0
    while (i<num_img):
        porosity = porosity_range[0] + np.random.rand()*(porosity_range[1] - porosity_range[0])
        img = generate_periodic_blobs(shape=(img_size,img_size), density=1-porosity, sigma=sigma,seed=start_seed+num_images_made)
        px, py, labeled = detect_percolation(img)
        percolates = px and py
        if percolates:
            filled = fill_non_percolating_fluid(img)
            images[i] = img
            images_filled[i] = filled
            porosities[i] = porosity
            i+=1
        num_images_made+=1
        print(f"Number of images made: {num_images_made}, kept: {i}")
    np.save(os.path.join(path,"test_images.npy"),images)
    np.save(os.path.join(path,"test_images_filled.npy"),images_filled)
    np.save(os.path.join(path,"test_porosities.npy"),porosities)
    

import time
if __name__ == "__main__":
    # demo()
    num_images = 4_000
    img_size = 128
    porosity_range = [0.001,0.9]
    start = time.time()
    porous_media_generation(num_img=num_images,
                            img_size=img_size,
                            porosity_range=porosity_range)
    
    
