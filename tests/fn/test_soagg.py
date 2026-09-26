"""soagg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.soagg import soagg


def test_soagg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        soagg()
