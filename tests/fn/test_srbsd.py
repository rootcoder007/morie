"""srbsd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srbsd import srbsd


def test_srbsd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srbsd()
