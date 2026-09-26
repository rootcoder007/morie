"""splogit is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.splogit import splogit


def test_splogit_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        splogit(y=None, X=None, W=None)
