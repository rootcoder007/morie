"""kdenf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kdenf import kdenf


def test_kdenf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kdenf()
