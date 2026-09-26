"""bayfin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bayfin import finite_mixture


def test_bayfin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        finite_mixture(y=None, K=None)
