"""nmplr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmplr import leg_polarize


def test_nmplr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        leg_polarize(data=None)
