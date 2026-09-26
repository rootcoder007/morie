"""gpvfe is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gpvfe import gpvfe


def test_gpvfe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gpvfe()
