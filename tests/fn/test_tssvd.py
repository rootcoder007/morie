"""tssvd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssvd import tssvd


def test_tssvd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssvd()
