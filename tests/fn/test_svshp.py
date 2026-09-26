"""svshp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svshp import shapley_spatial


def test_svshp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        shapley_spatial(data=None)
