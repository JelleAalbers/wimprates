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
    """
    This function creates a Cox Migdal model for a given element.

    Parameters:
    - element (str): The element for which the Cox Migdal model is created.
    - **kwargs: Additional keyword arguments for loading probabilities and total probabilities.

    Returns:
    - material: The Cox Migdal material object.

    Example usage:
    cox_migdal_model("carbon", arg1=value1, arg2=value2)

    Note: The Cox's model assumes that the main process is running in its root directory and uses
        relative paths. Therefore, we need to switch the working directory to the root of the package
        when computing the interpolators. 
        This wrapper function changes the working directory temporarily, instantiates the Migdal class, 
        and then resets the working directory back to its original state.
    """
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
