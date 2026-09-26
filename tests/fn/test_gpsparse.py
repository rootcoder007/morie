"""gpsparse is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gpsparse import gpsparse


def test_gpsparse_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gpsparse()
