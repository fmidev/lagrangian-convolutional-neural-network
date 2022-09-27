"""Methods for reading and writing files from/to disk."""

import gzip
from matplotlib.pyplot import imread
import numpy as np


def get_method(name):
    """Return the requested reader method. The only currently implemented
    option is 'pgm_gzip'."""
    if name == "pgm_gzip":
        return read_fmi_pgm_composite
    else:
        raise ValueError(f"unknown importer {name} requested, must be 'pgm_gzip")


def read_fmi_pgm_composite(filename, no_data_value=-32):
    """Read gzipped PGM composite from the FMI radar archive.

    Parameters
    ----------
    filename : str
        The name of the file to read.
    no_data_value : float
        The value that is assigned for pixels with missing data.

    Returns
    -------
    out : ndarray
        The composite read from the archive.
    """
    data = imread(gzip.open(filename, "r"))
    mask = data == 255
    data = data.astype(np.float64)
    data = (data - 64.0) / 2.0
    data[mask] = no_data_value

    return data
