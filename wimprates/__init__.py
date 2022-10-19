__version__ = '0.4.1'

from packaging import version
if version.parse('0.4.1') < version.parse(__version__) < version.parse('0.6.0'):
    # Remove this warning at some point
    import warnings
    warnings.warn(
        'Default WIMP parameters are changed in accordance with '
        'https://arxiv.org/abs/2105.00599 (github.com/JelleAalbers/wimprates/pull/14)')

from .utils import *
from .halo import *
from .elastic_nr import *
from .bremsstrahlung import *
from .migdal import *
from .electron import *
from .summary import *

