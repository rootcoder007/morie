"""sitrans is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sitrans import sitrans


def test_sitrans_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sitrans()
