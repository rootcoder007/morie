"""intinf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.intinf import interaction_information


def test_intinf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        interaction_information(pxyz=None)
