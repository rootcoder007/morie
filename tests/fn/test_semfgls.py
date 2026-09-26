"""semfgls is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.semfgls import semfgls


def test_semfgls_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        semfgls(y=None, X=None, W=None)
