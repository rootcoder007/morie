"""cldbi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cldbi import cldbi


def test_cldbi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cldbi()
