"""bbtrs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bbtrs import blackbox_transpose


def test_bbtrs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        blackbox_transpose()
