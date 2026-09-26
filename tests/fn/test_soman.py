"""soman is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.soman import soman


def test_soman_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        soman()
