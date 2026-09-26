"""gppred is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gppred import gppred


def test_gppred_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gppred()
