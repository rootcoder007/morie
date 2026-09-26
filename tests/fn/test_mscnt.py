"""mscnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mscnt import continuity


def test_mscnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        continuity(data=None)
