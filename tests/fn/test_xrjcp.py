"""xrjcp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrjcp import join_count_perm


def test_xrjcp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        join_count_perm(data=None)
