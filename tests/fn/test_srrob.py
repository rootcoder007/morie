"""srrob is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srrob import srrob


def test_srrob_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srrob()
