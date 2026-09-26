"""rclag is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rclag import rclag


def test_rclag_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rclag()
