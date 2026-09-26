"""percs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.percs import percs


def test_percs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        percs()
