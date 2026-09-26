"""sbprt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sbprt import sbprt


def test_sbprt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sbprt()
