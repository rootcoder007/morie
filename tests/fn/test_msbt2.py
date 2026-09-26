"""msbt2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msbt2 import mds_bootstrap


def test_msbt2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mds_bootstrap(X=None)
