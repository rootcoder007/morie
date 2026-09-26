"""ppswos is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppswos import pps_without_replacement


def test_ppswos_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pps_without_replacement(sizes=None, n=None)
