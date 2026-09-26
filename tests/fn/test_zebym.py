"""zebym is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zebym import bym_model


def test_zebym_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bym_model(data=None)
