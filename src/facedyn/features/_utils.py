"""Small helpers shared across more than one module in this subpackage."""

from __future__ import annotations

import numpy as np


def is_constant(x: np.ndarray) -> bool:
    """Whether `x` is constant to within floating-point tolerance.

    Not ``x.std() == 0``. Constant AU series (and features computed from
    them) still carry about 1e-30 of upstream floating-point noise, so an
    exact test reports them as varying and lets near-zero denominators
    through.
    """
    if len(x) == 0:
        return True
    return bool(np.allclose(x, x[0], rtol=1e-8, atol=1e-12, equal_nan=False))
