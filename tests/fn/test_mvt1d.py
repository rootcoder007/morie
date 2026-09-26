"""mvt1d is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mvt1d import mvt1d


def test_mvt1d_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mvt1d()
