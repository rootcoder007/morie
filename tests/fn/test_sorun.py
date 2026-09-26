"""sorun is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sorun import sorun


def test_sorun_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sorun()
