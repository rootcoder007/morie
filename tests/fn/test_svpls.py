"""svpls is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svpls import party_sorting


def test_svpls_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        party_sorting(data=None)
