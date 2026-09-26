"""gdemg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdemg import gdemg


def test_gdemg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdemg()
