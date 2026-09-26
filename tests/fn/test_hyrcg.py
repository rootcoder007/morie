"""hyrcg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyrcg import hyrcg


def test_hyrcg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyrcg()
