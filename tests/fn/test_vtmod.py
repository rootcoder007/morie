"""vtmod is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vtmod import vtmod


def test_vtmod_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vtmod()
