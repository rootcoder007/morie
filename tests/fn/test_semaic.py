"""semaic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.semaic import semaic


def test_semaic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        semaic(ll=None, k=None, n=None)
