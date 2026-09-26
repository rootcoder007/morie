"""sgtrnh is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgtrnh import sgt_randic_index


def test_sgtrnh_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgt_randic_index(A=None)
