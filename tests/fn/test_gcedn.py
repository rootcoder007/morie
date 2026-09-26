"""gcedn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcedn import gcedn


def test_gcedn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcedn()
