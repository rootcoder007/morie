"""wqtbn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqtbn import wqtbn


def test_wqtbn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqtbn()
