"""nmalp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmalp import alpha_nom_post


def test_nmalp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        alpha_nom_post(data=None)
