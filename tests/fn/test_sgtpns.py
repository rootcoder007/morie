"""sgtpns is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgtpns import sgt_perron_frobenius


def test_sgtpns_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgt_perron_frobenius(M=None)
