"""afslc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afslc import afslc


def test_afslc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afslc()
