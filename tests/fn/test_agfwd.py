"""agfwd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agfwd import agfwd


def test_agfwd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agfwd(options=None, setter_ideal=None, reversion=None)
