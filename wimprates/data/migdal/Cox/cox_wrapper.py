"""
The paths in the code in Peter Cox's package assume the working directory to be
the root of its package. With this wrapper we change the working dir when computing
the interpolators, then returning the Migdal class instance once they've been instantiated.
The working directory is then reset.
"""

import os
import sys

import wimprates as wr

from .cox_submodule.Migdal import Migdal

export, __all__ = wr.exporter()


@export
def cox_migdal_model(element: str, **kwargs):
    original_cwd = os.getcwd()

    try:
        migdal_directory = os.path.join(os.path.dirname(__file__), "cox_submodule")
        os.chdir(migdal_directory)

        material = Migdal(element)
        material.load_probabilities(**kwargs)
        material.load_total_probabilities(**kwargs)
    finally:
        os.chdir(original_cwd)

    return material
