"""zebot is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zebot import bayes_outbreak


def test_zebot_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bayes_outbreak(data=None)
