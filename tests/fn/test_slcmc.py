"""slcmc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.slcmc import slice_sampler


def test_slcmc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        slice_sampler(log_p=None, x0=None, width=None)
