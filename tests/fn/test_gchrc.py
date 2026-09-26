"""gchrc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gchrc import gchrc


def test_gchrc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gchrc()
