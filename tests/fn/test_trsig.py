"""trsig is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trsig import trsig


def test_trsig_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trsig()
