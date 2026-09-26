"""pldyn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pldyn import pldyn


def test_pldyn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pldyn()
