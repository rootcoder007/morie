"""dklrf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dklrf import dklrf


def test_dklrf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dklrf()
