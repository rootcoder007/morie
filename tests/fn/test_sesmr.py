"""sesmr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sesmr import sesmr


def test_sesmr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sesmr()
