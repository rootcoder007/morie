"""gatri is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gatri import gatri


def test_gatri_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gatri()
