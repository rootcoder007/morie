"""soclm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.soclm import soclm


def test_soclm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        soclm()
