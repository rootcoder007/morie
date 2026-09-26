"""nmala is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmala import alpha_nom_accept


def test_nmala_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        alpha_nom_accept(data=None)
