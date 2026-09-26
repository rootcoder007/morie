"""svmpc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svmpc import multiparty_comp


def test_svmpc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        multiparty_comp(data=None)
