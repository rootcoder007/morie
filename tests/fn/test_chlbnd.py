"""chlbnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chlbnd import chlbnd


def test_chlbnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chlbnd()
