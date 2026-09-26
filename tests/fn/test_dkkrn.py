"""dkkrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkkrn import dkkrn


def test_dkkrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkkrn()
