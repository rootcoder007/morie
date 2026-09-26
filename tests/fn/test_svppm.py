"""svppm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svppm import party_manifesto


def test_svppm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        party_manifesto(data=None)
