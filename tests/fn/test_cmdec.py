"""cmdec is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cmdec import cmdec


def test_cmdec_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cmdec()
