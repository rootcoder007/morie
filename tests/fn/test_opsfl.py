"""opsfl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opsfl import opsfl


def test_opsfl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opsfl()
