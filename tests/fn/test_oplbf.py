"""oplbf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.oplbf import oplbf


def test_oplbf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        oplbf()
