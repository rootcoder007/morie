"""enfld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enfld import enfld


def test_enfld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enfld()
