"""wqsec is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqsec import wqsec


def test_wqsec_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqsec()
