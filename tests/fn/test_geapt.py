"""geapt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.geapt import geapt


def test_geapt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        geapt()
