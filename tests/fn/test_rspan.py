"""rspan is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rspan import rspan


def test_rspan_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rspan()
