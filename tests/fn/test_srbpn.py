"""srbpn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srbpn import srbpn


def test_srbpn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srbpn()
