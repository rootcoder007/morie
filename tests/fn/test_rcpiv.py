"""rcpiv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rcpiv import rcpiv


def test_rcpiv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rcpiv()
