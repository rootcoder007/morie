"""clcph is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clcph import clcph


def test_clcph_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clcph()
