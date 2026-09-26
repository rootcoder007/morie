"""nmoc2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmoc2 import oc_2d


def test_nmoc2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        oc_2d(data=None)
