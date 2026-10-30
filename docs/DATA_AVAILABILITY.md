# Data availability and public-data policy

This GitHub package does not distribute 3D data volumes because of their size.
The following are excluded even when they are experimental outputs:

- paired GT/SIRT RAW files;
- reconstructed prediction RAW files;
- NumPy, NIfTI, MHA/MHD, HDF5, or similar multidimensional arrays;
- checkpoints and model weights.

The public `results/` folder contains only lightweight derived artifacts such
as CSV/JSON tables, PNG/PDF figures, and the final experiment report. These are
sufficient to inspect reported metrics without redistributing the volumes.

If sample data are later published through an external archive, use a separate
versioned release and provide checksums, data-generation settings, license, and
a stable DOI/URL. Do not commit the volumes directly to the Git repository.
