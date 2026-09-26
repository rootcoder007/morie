"""somag is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.somag import somag


def test_somag_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        somag()
