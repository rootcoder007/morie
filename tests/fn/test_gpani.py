"""gpani is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gpani import gpani


def test_gpani_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gpani()
