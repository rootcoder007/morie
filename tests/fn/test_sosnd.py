"""sosnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sosnd import sosnd


def test_sosnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sosnd()
