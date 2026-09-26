"""enpm2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enpm2 import enpm2


def test_enpm2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enpm2()
