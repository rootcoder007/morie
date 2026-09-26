"""soslt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.soslt import soslt


def test_soslt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        soslt()
