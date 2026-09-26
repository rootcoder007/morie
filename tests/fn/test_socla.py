"""socla is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.socla import socla


def test_socla_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        socla()
