"""diffs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.diffs import symbolic_diff


def test_diffs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        symbolic_diff(expr=None, x=None)
