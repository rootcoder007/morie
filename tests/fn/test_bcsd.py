"""bcsd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bcsd import bcsd_downscaling


def test_bcsd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bcsd_downscaling(gcm=None, obs=None)
