"""mtdns is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtdns import mtdns


def test_mtdns_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtdns()
