"""hyrpn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyrpn import hyrpn


def test_hyrpn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyrpn()
