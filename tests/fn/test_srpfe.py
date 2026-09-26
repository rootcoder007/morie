"""srpfe is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srpfe import srpfe


def test_srpfe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srpfe()
