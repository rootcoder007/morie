"""afrnf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afrnf import afrnf


def test_afrnf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afrnf()
