"""secla is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.secla import secla


def test_secla_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        secla()
