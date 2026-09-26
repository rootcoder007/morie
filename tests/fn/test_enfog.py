"""enfog is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enfog import enfog


def test_enfog_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enfog()
