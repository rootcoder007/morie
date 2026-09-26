"""tbexp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tbexp import tbexp


def test_tbexp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tbexp()
