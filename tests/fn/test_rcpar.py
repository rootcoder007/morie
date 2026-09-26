"""rcpar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rcpar import rcpar


def test_rcpar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rcpar()
