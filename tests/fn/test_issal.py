"""issal is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.issal import issal


def test_issal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        issal()
