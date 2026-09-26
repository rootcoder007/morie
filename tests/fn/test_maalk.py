"""maalk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maalk import maalk


def test_maalk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maalk()
