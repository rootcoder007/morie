"""gnssig is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gnssig import gnssig


def test_gnssig_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gnssig(resid=None, n=None)
