"""fosucc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fosucc import fosucc


def test_fosucc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fosucc()
