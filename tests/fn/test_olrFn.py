"""olrFn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.olrFn import outgoing_longwave


def test_olrFn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        outgoing_longwave(T_s=None)
