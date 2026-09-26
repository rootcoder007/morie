"""cttdis is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cttdis import ctt_discrimination


def test_cttdis_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ctt_discrimination(X=None)
