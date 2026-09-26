"""sbnsh is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sbnsh import sbnsh


def test_sbnsh_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sbnsh()
