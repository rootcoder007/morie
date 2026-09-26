"""sawdn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawdn import sawdn


def test_sawdn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawdn()
