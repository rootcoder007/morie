"""hubsau is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hubsau import hits_hub_authority


def test_hubsau_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hits_hub_authority(G=None)
