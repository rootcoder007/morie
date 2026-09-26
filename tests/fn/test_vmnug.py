"""vmnug is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmnug import vmnug


def test_vmnug_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmnug()
