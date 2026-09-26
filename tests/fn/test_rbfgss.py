"""rbfgss is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rbfgss import rbfgss


def test_rbfgss_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rbfgss()
