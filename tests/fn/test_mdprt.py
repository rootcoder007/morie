"""mdprt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mdprt import mdprt


def test_mdprt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mdprt()
