"""secln is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.secln import secln


def test_secln_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        secln()
