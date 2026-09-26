"""raindq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.raindq import rainbow_dqn


def test_raindq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rainbow_dqn(env=None)
