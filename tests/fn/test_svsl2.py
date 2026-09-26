"""svsl2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svsl2 import salience_2issue


def test_svsl2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        salience_2issue(data=None)
