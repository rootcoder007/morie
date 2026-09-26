"""ptmor is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptmor import pp_morisita


def test_ptmor_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pp_morisita(data=None)
