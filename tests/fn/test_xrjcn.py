"""xrjcn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrjcn import join_count


def test_xrjcn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        join_count(data=None)
