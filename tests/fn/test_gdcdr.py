"""gdcdr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdcdr import gdcdr


def test_gdcdr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdcdr()
