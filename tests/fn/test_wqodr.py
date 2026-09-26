"""wqodr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqodr import wqodr


def test_wqodr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqodr()
