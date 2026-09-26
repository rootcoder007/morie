"""ptpcf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptpcf import pair_corr_fn


def test_ptpcf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pair_corr_fn(data=None)
