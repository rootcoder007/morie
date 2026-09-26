"""gdctz is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdctz import gdctz


def test_gdctz_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdctz()
