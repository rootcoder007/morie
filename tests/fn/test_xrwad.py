"""xrwad is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrwad import w_adaptive


def test_xrwad_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        w_adaptive(data=None)
