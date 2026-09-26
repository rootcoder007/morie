"""sqzext is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sqzext import squeeze_excite


def test_sqzext_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        squeeze_excite(x=None, reduction=None)
