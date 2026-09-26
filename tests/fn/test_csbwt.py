"""csbwt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csbwt import csbwt


def test_csbwt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csbwt()
