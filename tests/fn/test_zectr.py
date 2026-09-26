"""zectr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zectr import contact_trace_sp


def test_zectr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        contact_trace_sp(data=None)
