"""xrfgt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrfgt import getis_filter


def test_xrfgt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        getis_filter(data=None)
