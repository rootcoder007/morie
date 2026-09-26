"""elplm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.elplm import elplm


def test_elplm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        elplm()
