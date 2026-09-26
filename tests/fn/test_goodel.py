"""goodel is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.goodel import goodel


def test_goodel_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        goodel()
