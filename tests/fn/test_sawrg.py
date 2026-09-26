"""sawrg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawrg import sawrg


def test_sawrg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawrg()
