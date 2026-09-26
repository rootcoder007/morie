"""sgtest is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgtest import sgt_estrada_index


def test_sgtest_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgt_estrada_index(A=None)
