"""sosal is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sosal import sosal


def test_sosal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sosal()
