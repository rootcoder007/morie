"""rsfus is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsfus import rsfus


def test_rsfus_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsfus()
