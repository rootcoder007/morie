"""kgsmw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgsmw import sk_weights


def test_kgsmw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sk_weights(data=None)
