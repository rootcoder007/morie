"""gedur is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gedur import gedur


def test_gedur_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gedur()
