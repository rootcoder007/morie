"""agsim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agsim import agsim


def test_agsim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agsim()
