"""tbsph is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tbsph import tbsph


def test_tbsph_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tbsph()
