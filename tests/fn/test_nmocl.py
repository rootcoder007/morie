"""nmocl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmocl import oc_cutline


def test_nmocl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        oc_cutline(data=None)
