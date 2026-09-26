"""ptmrk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptmrk import marked_pp


def test_ptmrk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        marked_pp(data=None)
