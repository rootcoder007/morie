"""mtssm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtssm import mtssm


def test_mtssm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtssm()
