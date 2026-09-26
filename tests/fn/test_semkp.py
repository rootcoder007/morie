"""semkp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.semkp import semkp


def test_semkp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        semkp(y=None, X=None, W=None)
