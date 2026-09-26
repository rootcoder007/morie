"""tbnbnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tbnbnd import tbnbnd


def test_tbnbnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tbnbnd()
