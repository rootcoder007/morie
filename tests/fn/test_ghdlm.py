"""ghdlm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghdlm import ghdlm


def test_ghdlm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghdlm()
