"""hsmlt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hsmlt import hsmlt


def test_hsmlt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hsmlt()
