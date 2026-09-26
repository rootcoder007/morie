"""optlb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.optlb import optlb


def test_optlb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        optlb()
