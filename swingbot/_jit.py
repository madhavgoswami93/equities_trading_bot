"""Optional numba acceleration.

Everything runs without numba (pure Python fallback), just ~50-100x slower in
the simulation loop. Set SWINGBOT_NO_JIT=1 to force the fallback (handy when
stepping through the kernel in a debugger).
"""

import os

try:
    if os.environ.get("SWINGBOT_NO_JIT"):
        raise ImportError
    from numba import njit  # type: ignore

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


__all__ = ["HAVE_NUMBA", "njit"]
