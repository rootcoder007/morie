"""msshp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msshp import shepard_diag


def test_msshp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        shepard_diag(data=None)
