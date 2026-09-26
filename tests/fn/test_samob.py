"""samob is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.samob import samob


def test_samob_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        samob()
