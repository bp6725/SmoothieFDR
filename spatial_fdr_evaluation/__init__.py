"""
Spatial FDR Evaluation Framework
=================================

A framework for evaluating False Discovery Rate (FDR) control methods
that leverage spatial structure through Reproducing Kernel Hilbert Spaces (RKHS).
"""

__version__ = "0.1.0"
__author__ = "Binyamin Perets"

from . import data
from . import methods
from . import evaluation
from . import visualization
from . import utils

__all__ = [
    "data",
    "methods", 
    "evaluation",
    "visualization",
    "utils"
]
