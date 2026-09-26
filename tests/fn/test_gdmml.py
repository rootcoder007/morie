"""gdmml is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdmml import gdmml


def test_gdmml_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdmml()
