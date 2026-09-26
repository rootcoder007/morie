"""dkssm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkssm import dkssm


def test_dkssm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkssm()
