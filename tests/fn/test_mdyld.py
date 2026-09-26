"""mdyld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mdyld import mdyld


def test_mdyld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mdyld()
