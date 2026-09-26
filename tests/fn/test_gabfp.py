"""gabfp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gabfp import gabfp


def test_gabfp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gabfp()
