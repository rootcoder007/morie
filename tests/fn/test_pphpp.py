"""pphpp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pphpp import pphpp


def test_pphpp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pphpp()
