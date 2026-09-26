"""afvwc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afvwc import afvwc


def test_afvwc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afvwc()
