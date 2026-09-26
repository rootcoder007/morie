"""dkglb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkglb import dkglb


def test_dkglb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkglb()
