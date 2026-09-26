"""stprod is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stprod import stprod


def test_stprod_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stprod()
