"""rnslp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rnslp import rnslp


def test_rnslp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rnslp()
