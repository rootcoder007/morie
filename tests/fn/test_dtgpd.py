"""dtgpd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtgpd import dtgpd


def test_dtgpd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtgpd()
