"""sbrub is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sbrub import sbrub


def test_sbrub_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sbrub()
