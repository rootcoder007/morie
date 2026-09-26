"""hybre is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hybre import hybre


def test_hybre_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hybre()
