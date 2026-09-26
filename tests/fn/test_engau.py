"""engau is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.engau import engau


def test_engau_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        engau()
