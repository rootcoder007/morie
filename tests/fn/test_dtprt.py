"""dtprt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtprt import dtprt


def test_dtprt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtprt()
