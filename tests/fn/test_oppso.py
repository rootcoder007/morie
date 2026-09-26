"""oppso is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.oppso import oppso


def test_oppso_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        oppso()
