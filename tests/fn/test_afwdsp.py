"""afwdsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afwdsp import afwdsp


def test_afwdsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afwdsp()
