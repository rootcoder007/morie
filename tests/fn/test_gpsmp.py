"""gpsmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gpsmp import gpsmp


def test_gpsmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gpsmp()
