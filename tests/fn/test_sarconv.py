"""sarconv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sarconv import sarconv


def test_sarconv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sarconv(W=None)
