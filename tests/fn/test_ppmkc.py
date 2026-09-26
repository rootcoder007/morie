"""ppmkc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppmkc import ppmkc


def test_ppmkc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppmkc()
