"""ptdgm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptdgm import diggle_test


def test_ptdgm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        diggle_test(data=None)
