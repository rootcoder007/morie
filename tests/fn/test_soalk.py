"""soalk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.soalk import soalk


def test_soalk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        soalk()
