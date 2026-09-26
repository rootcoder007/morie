"""stper is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stper import stper


def test_stper_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stper()
