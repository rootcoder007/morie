"""cdcol is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdcol import cdcol


def test_cdcol_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdcol()
