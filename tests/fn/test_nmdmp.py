"""nmdmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmdmp import party_diverge


def test_nmdmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        party_diverge(data=None)
