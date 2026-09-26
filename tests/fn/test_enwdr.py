"""enwdr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enwdr import enwdr


def test_enwdr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enwdr()
