"""ppenv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppenv import ppenv


def test_ppenv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppenv()
