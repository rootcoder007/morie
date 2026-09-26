"""srhm2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srhm2 import srhm2


def test_srhm2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srhm2()
