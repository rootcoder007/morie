"""fanocb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fanocb import fano_inequality


def test_fanocb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fano_inequality(pe=None, X_card=None)
