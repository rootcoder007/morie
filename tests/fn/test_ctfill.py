"""ctfill is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ctfill import ctfill


def test_ctfill_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ctfill()
