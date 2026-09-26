"""xrgns is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrgns import gns_ml


def test_xrgns_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gns_ml(data=None)
