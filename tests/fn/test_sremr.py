"""sremr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sremr import sremr


def test_sremr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sremr()
