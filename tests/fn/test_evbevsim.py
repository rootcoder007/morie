"""evbevsim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.evbevsim import evt_bv_evd_sim


def test_evbevsim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        evt_bv_evd_sim(alpha=None, n=None)
