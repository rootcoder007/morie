"""gnsml is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gnsml import gnsml


def test_gnsml_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gnsml(y=None, X=None, W=None)
