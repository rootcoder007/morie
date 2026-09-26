"""dtgev is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtgev import dtgev


def test_dtgev_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtgev()
