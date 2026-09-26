"""ghnod is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghnod import ghnod


def test_ghnod_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghnod()
