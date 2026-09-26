"""soph is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.soph import soph


def test_soph_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        soph()
