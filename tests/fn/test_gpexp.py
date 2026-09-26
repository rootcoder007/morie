"""gpexp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gpexp import gpexp


def test_gpexp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gpexp()
