"""lisgst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lisgst import local_getis_g


def test_lisgst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        local_getis_g(x=None, W=None)
