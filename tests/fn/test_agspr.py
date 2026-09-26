"""agspr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agspr import agspr


def test_agspr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agspr()
