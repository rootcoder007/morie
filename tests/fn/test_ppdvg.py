"""ppdvg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppdvg import ppdvg


def test_ppdvg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppdvg()
