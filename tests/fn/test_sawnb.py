"""sawnb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawnb import sawnb


def test_sawnb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawnb()
