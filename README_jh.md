# Note

## inference single-coil

The ground truth of this experiment directly use the ifft of single-coil k-space data, instead of the reconstruction_rss.
This is different with the setting in multi-coil experiment.

## Data process

The normalisation complex function also normalised the phase, which is different with the setting in multi-coil experiment! This will lead to different ZF results.

This is used in inference multi-coil experiment, also in the preprocessuing of single-coil data (NOT RELEASE).