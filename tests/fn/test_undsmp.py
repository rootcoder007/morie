"""undsmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.undsmp import undsmp


def test_undsmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        undsmp()
