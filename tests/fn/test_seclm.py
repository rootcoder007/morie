"""seclm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seclm import seclm


def test_seclm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seclm()
