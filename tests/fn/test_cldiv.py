"""cldiv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cldiv import cldiv


def test_cldiv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cldiv()
