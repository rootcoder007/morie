"""srswm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srswm import srswm


def test_srswm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srswm()
