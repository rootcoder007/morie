"""lonlat is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lonlat import lonlat


def test_lonlat_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lonlat()
