"""trprt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trprt import trprt


def test_trprt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trprt()
