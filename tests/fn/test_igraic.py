"""igraic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.igraic import igraic


def test_igraic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        igraic(ll=None, k=None, n=None)
