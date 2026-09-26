"""sacml is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sacml import sacml


def test_sacml_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sacml(y=None, X=None, W=None)
