"""idwani is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idwani import idwani


def test_idwani_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idwani()
