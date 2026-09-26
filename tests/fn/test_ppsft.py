"""ppsft is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppsft import ppsft


def test_ppsft_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppsft()
