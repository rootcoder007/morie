"""ctisoq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ctisoq import ctisoq


def test_ctisoq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ctisoq()
