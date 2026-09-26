"""zssgs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zssgs import seq_gauss_sim


def test_zssgs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seq_gauss_sim(data=None)
