"""spcphs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spcphs import spcphs


def test_spcphs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spcphs()
