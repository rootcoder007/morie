"""mcsec is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mcsec import mcmc_standard_error


def test_mcsec_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mcmc_standard_error(chains=None)
