"""dtghs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtghs import dtghs


def test_dtghs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtghs()
