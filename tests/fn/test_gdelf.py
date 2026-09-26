"""gdelf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdelf import gdelf


def test_gdelf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdelf()
