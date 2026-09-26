"""gdmig is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdmig import gdmig


def test_gdmig_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdmig()
