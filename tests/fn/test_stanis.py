"""stanis is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stanis import stanis


def test_stanis_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stanis()
