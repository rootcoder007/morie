"""vorone is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vorone import vorone


def test_vorone_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vorone()
