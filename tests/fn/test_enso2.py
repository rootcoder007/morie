"""enso2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enso2 import enso2


def test_enso2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enso2()
