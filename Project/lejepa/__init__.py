from . import lejepa as _inner
from .lejepa import *  # noqa: F401,F403

import sys

# Re-export inner subpackages so existing imports like
# `from lejepa.univariate import EppsPulley` keep working.
sys.modules[__name__ + ".univariate"] = _inner.univariate
sys.modules[__name__ + ".multivariate"] = _inner.multivariate

__all__ = getattr(_inner, "__all__", [])
