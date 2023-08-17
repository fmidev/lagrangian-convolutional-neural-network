"""
Get conversion and transformation methods from pysteps

Bent Harnist (FMI) 2022
"""
from functools import partial

import numpy as np
from pysteps.utils.conversion import to_rainrate, to_reflectivity
from pysteps.utils.transformation import dB_transform


def get_unit_conversion_method(name: str):
    """Get unit conversion method with specified name from PySTEPS.

    Args:
        name (str): unit conversion method name ("reflectivity"|"rainrate")

    Raises:
        ValueError: if given name is not defined.

    Returns:
        callable: unit conversion method.
    """
    if name == "reflectivity":
        return to_reflectivity
    elif name == "rainrate":
        return to_rainrate
    else:
        raise ValueError(
            f"unit conversion method name {name} undefined. Choices: ['reflectivity', rainrate']"
        )


def get_transformation_method(name: str):
    """Get data transformation method with specified name from PySTEPS.

    Args:
        name (str): unit conversion method name ("db"|"db_inverse")

    Raises:
        ValueError: if given name is not defined.

    Returns:
        callable: data transformation method.
    """
    if name == "db":
        return dB_transform
    elif name == "db_inverse":
        return partial(dB_transform, inverse=True)
    else:
        raise ValueError(
            f"transformation method name {name} undefined. Choices: ['db', db_inverse']"
        )


def dbz_to_rainrate(
    data_dBZ: np.ndarray, zr_a: float = 223, zr_b: float = 1.53
) -> np.ndarray:
    """Convert a dbz reflectivity array to a rain rate estimate (mm/h).

    Args:
        data_dBZ (np.ndarray): dbz reflectivity array
        zr_a (float, optional): A Z-R relationship coefficient. Defaults to 223 (FMI data).
        zr_b (float, optional): b Z-R relationship coefficient. Defaults to 1.53 (FMI data).

    Returns:
        np.ndarray: rain rate data (mm/h)
    """
    data = 10 ** (data_dBZ * 0.1)  # dB - inverse transform -> Z
    data = (data / zr_a) ** (1 / zr_b)  # Z -> R
    return data


def rainrate_to_dbz(
    R: np.ndarray,
    zr_a: float = 223,
    zr_b: float = 1.53,
    thresh: float = 0.1,
    zerovalue=-32,
) -> np.ndarray:
    """Convert a rain rate estimate to a dbz reflectivity array.

    Args:
        R (np.ndarray): rain rate array (mm/h)
        zr_a (float, optional): A Z-R relationship coefficient. Defaults to 223 (FMI data).
        zr_b (float, optional): b Z-R relationship coefficient. Defaults to 1.53 (FMI data).
        thresh (float, optional): threshold for observable rain rate (mm/h). Defaults to 0.1.
        zerovalue (int, optional): value for reflectivity of data under threshold of
            observable rain. Defaults to -32.

    Returns:
        np.ndarray: reflectivity data (dbz)
    """
    zeros = R < thresh
    Z = zr_a * R**zr_b  # R -> Z
    Z = 10 * np.log10(Z)  # Z -> dBZ
    Z[zeros] = zerovalue  # fill values under threshold with zerovalues
    return Z