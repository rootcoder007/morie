"""stexp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stexp import stexp


def test_stexp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stexp()
