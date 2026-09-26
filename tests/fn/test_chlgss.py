"""chlgss is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chlgss import chlgss


def test_chlgss_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chlgss()
