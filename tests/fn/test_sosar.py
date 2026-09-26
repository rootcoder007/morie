"""sosar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sosar import sosar


def test_sosar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sosar()
