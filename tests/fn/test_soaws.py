"""soaws is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.soaws import soaws


def test_soaws_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        soaws()
