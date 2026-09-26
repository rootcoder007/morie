"""vtmar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vtmar import vtmar


def test_vtmar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vtmar()
