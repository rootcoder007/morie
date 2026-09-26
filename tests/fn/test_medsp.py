"""medsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.medsp import medsp


def test_medsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        medsp()
