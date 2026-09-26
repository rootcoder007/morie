"""nmocc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmocc import oc_classify


def test_nmocc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        oc_classify(data=None)
