"""tvdist is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tvdist import total_variation_distance


def test_tvdist_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        total_variation_distance(y=None, p=None, q=None)
