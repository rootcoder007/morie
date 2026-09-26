"""ptenv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptenv import pp_envelope


def test_ptenv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pp_envelope(data=None)
