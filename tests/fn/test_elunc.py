"""elunc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.elunc import elunc


def test_elunc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        elunc()
