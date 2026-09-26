"""mdrad is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mdrad import mdrad


def test_mdrad_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mdrad()
