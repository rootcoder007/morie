"""gemfi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gemfi import gemfi


def test_gemfi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gemfi()
