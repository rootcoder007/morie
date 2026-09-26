"""gdcmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdcmp import gdcmp


def test_gdcmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdcmp()
