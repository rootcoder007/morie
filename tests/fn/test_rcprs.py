"""rcprs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rcprs import rcprs


def test_rcprs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rcprs()
