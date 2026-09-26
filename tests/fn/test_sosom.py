"""sosom is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sosom import sosom


def test_sosom_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sosom()
