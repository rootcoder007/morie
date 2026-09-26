"""tqmsb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tqmsb import turboquant_mse_distortion_bound


def test_tqmsb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        turboquant_mse_distortion_bound(bits=None)
