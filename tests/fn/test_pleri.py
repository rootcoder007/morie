"""pleri is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pleri import pleri


def test_pleri_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pleri()
