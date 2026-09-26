"""vime is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vime import empirical_orthogonal_func


def test_vime_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        empirical_orthogonal_func(X=None)
