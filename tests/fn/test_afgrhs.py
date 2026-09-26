"""afgrhs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afgrhs import afgrhs


def test_afgrhs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afgrhs()
