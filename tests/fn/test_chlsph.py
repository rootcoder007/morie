"""chlsph is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chlsph import chlsph


def test_chlsph_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chlsph()
