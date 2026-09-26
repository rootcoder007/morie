"""svblp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svblp import bliss_point


def test_svblp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bliss_point(data=None)
