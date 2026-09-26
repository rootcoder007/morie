"""pplgc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pplgc import pplgc


def test_pplgc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pplgc()
