"""srbml is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srbml import srbml


def test_srbml_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srbml()
