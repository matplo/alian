import os
from collections import defaultdict

import numpy as np
import yaml


def read_yaml(f):
    with open(f) as stream:
        cfg = yaml.safe_load(stream)
    return cfg

def nested_dict():
    return defaultdict(nested_dict)

ndict = nested_dict

def dict_to_ndict(d):
    ndict = nested_dict()
    for k, v in d.items():
        if isinstance(v, dict):
            ndict[k] = dict_to_ndict(v)
        else:
            ndict[k] = v
    return ndict

def ndict_to_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ndict_to_dict(v)
    return dict(d)

def linbins(xmin, xmax, nbins):
    return np.linspace(xmin, xmax, nbins + 1)

def logbins(xmin, xmax, nbins):
    return np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)

def delta_R(p1, p2):
    """Calculate angular distance with pseudorapidity (rather than rapidity)."""
    return np.sqrt(p1.delta_phi_to(p2) ** 2 + (p1.eta() - p2.eta()) ** 2)

def delta_phi_limit(dphi):
    """Limit a given angle between -pi and pi."""
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi < -np.pi:
        dphi += 2 * np.pi
    return dphi

def is_slurm():
    """Check if we are in a Slurm job / on a compute node."""
    if "SLURM_SUBMIT_DIR" in os.environ or 'HOST' not in os.environ:
        # check for Slurm environment variables or an unset HOST variable
        return True
    elif 'login' in os.environ['HOST']:
        # check for login node for HOST
        return False
    else:
        # if unparsable, default to Slurm configuration (safer)
        return True