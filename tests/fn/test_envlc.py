"""envlc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.envlc import envlc


def test_envlc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        envlc()
