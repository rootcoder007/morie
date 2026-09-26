"""gebgt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gebgt import gebgt


def test_gebgt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gebgt()
