"""dtmap is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtmap import dtmap


def test_dtmap_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtmap()
