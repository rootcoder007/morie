"""seclk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seclk import seclk


def test_seclk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seclk()
