"""svprc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svprc import svprc


def test_svprc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svprc()
