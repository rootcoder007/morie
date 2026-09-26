"""mtmsd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtmsd import mtmsd


def test_mtmsd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtmsd()
