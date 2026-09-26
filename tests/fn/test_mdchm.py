"""mdchm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mdchm import mdchm


def test_mdchm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mdchm()
