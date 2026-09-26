"""igrbic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.igrbic import igrbic


def test_igrbic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        igrbic(ll=None, k=None, n=None)
