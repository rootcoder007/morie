"""floRate is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.floRate import flow_duration


def test_floRate_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        flow_duration(Q=None)
