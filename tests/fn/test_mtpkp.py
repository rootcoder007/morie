"""mtpkp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtpkp import mtpkp


def test_mtpkp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtpkp()
