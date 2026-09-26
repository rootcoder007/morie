"""hybod is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hybod import hybod


def test_hybod_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hybod()
