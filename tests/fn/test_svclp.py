"""svclp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svclp import cut_plane


def test_svclp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cut_plane(data=None)
