"""sgmean is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgmean import sgmean


def test_sgmean_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgmean()
