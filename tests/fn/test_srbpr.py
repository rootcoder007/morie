"""srbpr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srbpr import srbpr


def test_srbpr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srbpr()
