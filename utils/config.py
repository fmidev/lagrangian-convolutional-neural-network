import yaml
from attrdict import AttrDict


def load_config(file):
    """Load configuration from YAML file.

    Parameters
    ----------
    file : str
        Path to configuration file.

    Returns
    -------
    conf_dict : attrdict.AttrDict
        AttrDict containing configurations.

    """
    # read configuration files
    with open(file, "r") as f:
        conf_dict = AttrDict(yaml.safe_load(f))
    return conf_dict
