"""rsrfc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsrfc import rsrfc


def test_rsrfc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsrfc()
