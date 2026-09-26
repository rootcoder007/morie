"""gcfld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcfld import gcfld


def test_gcfld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcfld()
