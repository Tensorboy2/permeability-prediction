'''
data_pipeline_2.py

Module for running Lattice-Boltzmann simulations on 2d porous media and storing the resulting permeability tensor.
MPI compatible.
'''
import numpy as np
import os
from mpi4py import MPI
from lbm import big_LBM
path = os.path.dirname(__file__)

# out_dir = os.path.join(path, "test_4000_results")
# os.makedirs(out_dir, exist_ok=True)
# out_dir = os.path.join(path, "train_and_val")
# os.makedirs(out_dir, exist_ok=True)

def image_already_done(index, out_dir):
    return (
        os.path.exists(os.path.join(out_dir, f"u_x_{index:05d}.npy")) and
        os.path.exists(os.path.join(out_dir, f"u_y_{index:05d}.npy")) and
        os.path.exists(os.path.join(out_dir, f"k_{index:05d}.npy"))
    )

def save_result(index, u_x, u_y, k, out_dir):
    np.save(os.path.join(out_dir, f"u_x_{index:05d}.npy"), u_x)
    np.save(os.path.join(out_dir, f"u_y_{index:05d}.npy"), u_y)
    np.save(os.path.join(out_dir, f"k_{index:05d}.npy"), k)


def main_mpi_from_dir(output_prefix="validation", images_source=None, index_offset=0):
    '''
    MPI version of data pipeline for simulating and calculating permeability.
    '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    T = 10_000
    # backward-compatible handling for the misspelled arg
    if output_prefix is not None:
        output_prefix = output_prefix

    # Create correct output directory
    out_dir = os.path.join(path, output_prefix)
    os.makedirs(out_dir, exist_ok=True)

    # Determine image source: either a directory (default) or a single .npy file with all images
    if images_source is None:
        image_dir = os.path.join(path, "images_filled")
    else:
        image_dir = images_source

    # Determine mode and total number of images on rank 0
    if rank == 0:
        print("[INFO] Preparing image list for processing...")
        if os.path.isdir(image_dir):
            images_mode = 'dir'
            all_filenames = sorted(os.listdir(image_dir))
            all_filenames = [f for f in all_filenames if f.endswith(".npy")]
            N = len(all_filenames)
            print(f"[INFO] Found {N} image files in directory {image_dir}.")
        elif os.path.isfile(image_dir) and image_dir.endswith('.npy'):
            images_mode = 'npy_file'
            # Load only header to determine N
            tmp = np.load(image_dir, mmap_mode='r')
            N = tmp.shape[0]
            print(f"[INFO] Found consolidated .npy file {image_dir} with {N} images.")
            all_filenames = None
        else:
            raise FileNotFoundError(f"Image source {image_dir} not found or unsupported")
    else:
        images_mode = None
        all_filenames = None
        N = None

    # Broadcast essential information to all ranks
    images_mode = comm.bcast(images_mode, root=0)
    image_dir = comm.bcast(image_dir, root=0)
    N = comm.bcast(N, root=0)

    # Ensure even distribution across ranks:
    local_N = N // size
    remainder = N % size
    if rank < remainder:
        local_start = rank * (local_N + 1)
        local_end = local_start + local_N + 1
    else:
        local_start = rank * local_N + remainder
        local_end = local_start + local_N

    # Prepare chunks of work (either filenames or numeric indices) and scatter
    if rank == 0:
        chunks = []
        for r in range(size):
            if r < remainder:
                s = r * (local_N + 1)
                e = s + local_N + 1
            else:
                s = r * local_N + remainder
                e = s + local_N
            if images_mode == 'dir':
                chunks.append(all_filenames[s:e])
            else:
                # npy_file mode: assign numeric indices
                chunks.append(list(range(s, e)))
    else:
        chunks = None

    # Scatter the prepared chunks to each rank
    local_filenames = comm.scatter(chunks, root=0)

    # If using consolidated npy file, let each rank load the big array (mmap mode to reduce memory)
    big_array = None
    if images_mode == 'npy_file':
        big_array = np.load(image_dir, mmap_mode='r')

    for i, fname in enumerate(local_filenames):
        global_idx = index_offset + local_start + i

        if image_already_done(global_idx,out_dir=out_dir):
            print(f"[RANK {rank}] [SKIP] Image {global_idx+1}/{N} already processed.")
            continue

        print(f"\n[RANK {rank}] [INFO] Processing image {global_idx+1}/{N} (global_idx={global_idx})...")
        image_start = MPI.Wtime()

        # Load image depending on mode
        if images_mode == 'dir':
            image_path = os.path.join(image_dir, fname)
            image = np.load(image_path)
        else:
            # fname is numeric index in this mode
            idx = fname
            image = big_array[idx]

        print(f"[RANK {rank}] [INFO] Running LBM in x-direction...")
        u_x, k_xx, k_xy = big_LBM(image, T, 0)

        print(f"[RANK {rank}] [INFO] Running LBM in y-direction...")
        u_y, k_yx, k_yy = big_LBM(image, T, 1)

        k = np.array([[k_xx, k_xy], [k_yx, k_yy]])

        # Save results for this image index
        save_result(global_idx, u_x, u_y, k, out_dir)

        elapsed = MPI.Wtime() - image_start
        print(f"[RANK {rank}] [DONE] Image {global_idx+1} completed in {elapsed:.2f} seconds.")

    comm.Barrier()
    if rank == 0:
        print("[DONE] All ranks finished processing.")


if __name__ == '__main__':
    main_mpi_from_dir(output_prefix="train_and_val", images_source='data/images_filled.npy', index_offset=0)
    main_mpi_from_dir(output_prefix="test_4000", images_source='data/test_4000_images_filled.npy', index_offset=20000)