"""gemrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gemrn import gemrn


def test_gemrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gemrn()
