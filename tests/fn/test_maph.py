"""maph is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maph import maph


def test_maph_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maph()
