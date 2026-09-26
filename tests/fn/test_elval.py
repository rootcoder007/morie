"""elval is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.elval import elval


def test_elval_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        elval()
