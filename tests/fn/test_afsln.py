"""afsln is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afsln import afsln


def test_afsln_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afsln()
