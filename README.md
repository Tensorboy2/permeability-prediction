# Permeability Prediction
By Sigurd Sønvisen Vargdal

This repository contains code for generating data and training deep learning image methods on prediction of permeability.

The data generation involves: generating synthetic periodic porous medium with arbitrary geometry and running Lattice-Boltzmann simulations on the medium to obtain the respective *permeability* tensor.

The porous media is generated using a periodic version of the **binary_blobs** function from SciKit-Image. Each medium is required to percolate in both the x and y direction. We fill disconnected fluid clusters to obtain the image filled version of each medium. For flow simulations we use the Lattice-Boltzmann method with the $D2Q9$ lattice. With the obtained velocity field we compute the permeability using Darcy's law. 

To generate the data run:
```bash
python3 data/generate_images.py
```
Then generate the test images with:
```bash
python3 data/generate_test_images.py
```

The simulation is performed using MPI with one process per sample. So for 8 cores run:
```bash
mpirun -np 8 data/simulation_pipeline.py
```




