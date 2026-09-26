"""sdmml is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sdmml import sdmml


def test_sdmml_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sdmml(y=None, X=None, W=None)
