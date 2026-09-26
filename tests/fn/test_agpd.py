"""agpd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agpd import agpd


def test_agpd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agpd()
