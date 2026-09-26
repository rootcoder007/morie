"""svamn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svamn import amendment_seq


def test_svamn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        amendment_seq(data=None)
