"""tssre is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssre import tssre


def test_tssre_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssre()
