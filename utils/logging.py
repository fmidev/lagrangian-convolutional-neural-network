"""Utility functions related to logging."""
import logging
from pathlib import Path
from datetime import datetime


LEVELS = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def setup_logging(logconf):
    """Set up logging.

    Parameters
    ----------
    logconf : dict
        Dictionary of logging configurations.

    """
    date = datetime.now()
    if "filename" in logconf.keys() and logconf["filename"] is not None:
        path = Path(logconf["path"])
        # Make sure path exists
        path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=LEVELS[logconf["level"]],
            format=logconf["format"],
            datefmt=logconf["dateformat"],
            filename=path
            / logconf["filename"].format(
                year=date.year,
                month=date.month,
                day=date.day,
                hour=date.hour,
                minute=date.minute,
                second=date.second,
            ),
            filemode="w",
        )
    else:
        logging.basicConfig(
            level=LEVELS[logconf["level"]],
            format=logconf["format"],
            datefmt=logconf["dateformat"],
        )
