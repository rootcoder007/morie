"""cttdif is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cttdif import ctt_difficulty


def test_cttdif_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ctt_difficulty(X=None)
