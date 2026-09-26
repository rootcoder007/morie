"""focnpd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.focnpd import focnpd


def test_focnpd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        focnpd()
