"""gwraicc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gwraicc import gwraicc


def test_gwraicc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gwraicc(ll=None, k=None, n=None)
