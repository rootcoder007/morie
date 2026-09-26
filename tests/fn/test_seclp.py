"""seclp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seclp import seclp


def test_seclp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seclp()
