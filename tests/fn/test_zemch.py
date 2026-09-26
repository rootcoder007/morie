"""zemch is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zemch import maternal_child_map


def test_zemch_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maternal_child_map(data=None)
