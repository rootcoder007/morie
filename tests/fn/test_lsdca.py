"""lsdca is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lsdca import robust_lda


def test_lsdca_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        robust_lda(X=None, y=None)
